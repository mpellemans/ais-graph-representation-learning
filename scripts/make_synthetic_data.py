"""Generate a synthetic maritime dataset with the same schema as the licensed data.

The real AIS-derived dataset (S&P Global Sea-web) cannot be redistributed, so this
script produces a schema-faithful synthetic substitute: fake ports with coherent
regional geography, per-vessel port-visit trajectories, and route aggregates. Every
maritime entry point (preprocessing, DGI training, baselines, link prediction,
clustering, embedding analysis) runs end-to-end on the generated files. The numbers
it produces are NOT the paper's numbers; they only exercise the code paths.

The schemas mirror data/maritime/{nodes,edges,journeys}.parquet as documented in
docs/data_schema.md. The generator extends the synthetic fixtures used by
tests/test_maritime_preprocessing.py.

Usage:
    python scripts/make_synthetic_data.py                      # writes data/synthetic/
    python scripts/make_synthetic_data.py --n-ports 500 --n-journeys 60000
    python experiments/maritime/run_dgi_training.py --config configs/experiments/synthetic.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VESSEL_TYPES = [
    'Passenger', 'Container', 'Bulk Dry', 'Oil', 'Chemical',
    'Liquefied Gas', 'Ro-Ro Cargo', 'General Cargo',
]

# Region model: (sub_region, continent, country codes, center lat, center lon)
REGIONS = [
    ('Western Europe', 'Europe', ['NL', 'DE', 'FR', 'BE'], 51.0, 4.0),
    ('Southern Europe', 'Europe', ['ES', 'IT', 'GR'], 39.0, 12.0),
    ('South-Eastern Asia', 'Asia', ['SG', 'MY', 'ID', 'TH'], 3.0, 104.0),
    ('Eastern Asia', 'Asia', ['CN', 'JP', 'KR'], 32.0, 124.0),
    ('Northern America', 'Americas', ['US', 'CA', 'MX'], 35.0, -90.0),
    ('South America', 'Americas', ['BR', 'AR', 'CL'], -20.0, -50.0),
    ('Australia and New Zealand', 'Oceania', ['AU', 'NZ'], -30.0, 145.0),
]

TIME_START = pd.Timestamp('2022-02-01')
TIME_END = pd.Timestamp('2023-08-16')


def make_ports(n_ports, rng):
    """Assign ports to regions with coherent geography."""
    region_idx = rng.randint(0, len(REGIONS), n_ports)
    rows = []
    for i in range(n_ports):
        sub_region, continent, countries, clat, clon = REGIONS[region_idx[i]]
        country = countries[rng.randint(len(countries))]
        rows.append({
            'port_name': f'Port_{i:04d}',
            'latitude': float(np.clip(clat + rng.normal(0, 6.0), -75, 75)),
            'longitude': float(((clon + rng.normal(0, 10.0)) + 180) % 360 - 180),
            'country': country,
            'country_name': f'Country_{country}',
            'continent': continent,
            'sub_region': sub_region,
        })
    return pd.DataFrame(rows), region_idx


def make_preferred_destinations(n_ports, region_idx, rng, n_pref=14, p_same_region=0.7):
    """Per-port destination pool: mostly same-region plus global connections."""
    by_region = {}
    for r in range(len(REGIONS)):
        by_region[r] = np.where(region_idx == r)[0]
    prefs = []
    all_ports = np.arange(n_ports)
    for i in range(n_ports):
        same = by_region[region_idx[i]]
        same = same[same != i]
        pool = []
        for _ in range(n_pref):
            if len(same) > 0 and rng.random() < p_same_region:
                pool.append(int(same[rng.randint(len(same))]))
            else:
                j = int(all_ports[rng.randint(n_ports)])
                if j != i:
                    pool.append(j)
        pool = sorted(set(pool))
        if not pool:
            pool = [int((i + 1) % n_ports)]
        prefs.append(pool)
    return prefs


def make_journeys(ports_df, prefs, n_vessels, n_journeys, rng):
    """Per-vessel random-walk trajectories with increasing timestamps."""
    port_names = ports_df['port_name'].tolist()
    n_ports = len(port_names)
    total_seconds = (TIME_END - TIME_START).total_seconds()

    vessel_types = [VESSEL_TYPES[rng.randint(len(VESSEL_TYPES))] for _ in range(n_vessels)]
    steps_per_vessel = np.maximum(rng.poisson(n_journeys / n_vessels, n_vessels), 2)

    rows = []
    journey_id = 1
    for v in range(n_vessels):
        vessel_id = 1000000 + v
        current = rng.randint(n_ports)
        # Spread trajectory starts over the first weeks of the window
        t = TIME_START + pd.Timedelta(seconds=float(rng.uniform(0, 30 * 86400)))
        for _ in range(int(steps_per_vessel[v])):
            pool = prefs[current]
            nxt = pool[rng.randint(len(pool))]
            travel_h = float(rng.uniform(4, 240))
            dwell_h = float(rng.uniform(2, 120))
            arrival = t + pd.Timedelta(hours=travel_h)
            if (arrival - TIME_START).total_seconds() > total_seconds:
                break
            o_name, d_name = port_names[current], port_names[nxt]
            o_row = ports_df.iloc[current]
            d_row = ports_df.iloc[nxt]
            rows.append({
                'journey_id': journey_id,
                'vessel_id': vessel_id,
                'origin_port': o_name,
                'destination_port': d_name,
                'departure_time': t,
                'arrival_time': arrival,
                'origin_lat': o_row['latitude'],
                'origin_lon': o_row['longitude'],
                'destination_lat': d_row['latitude'],
                'destination_lon': d_row['longitude'],
                'origin_country': o_row['country'],
                'destination_country': d_row['country'],
                'draught_departure': float(rng.uniform(1, 10)),
                'draught_arrival': float(rng.uniform(1, 10)),
                'departure_port_visit_id': int(rng.randint(100000, 999999)),
                'arrival_port_visit_id': int(rng.randint(100000, 999999)),
                'departure_movement_type': 'Port Departure',
                'arrival_movement_type': 'Port Arrival',
                'is_complete_journey': True,
                'travel_time_hours': travel_h,
                'departure_year': t.year,
                'departure_month': t.month,
                'departure_day_of_week': t.dayofweek,
                'departure_hour': t.hour,
                'ShiptypeLevel3': vessel_types[v],
            })
            journey_id += 1
            current = nxt
            t = arrival + pd.Timedelta(hours=dwell_h)
    df = pd.DataFrame(rows)
    return df.sort_values(['vessel_id', 'departure_time']).reset_index(drop=True)


def entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def make_edges(journeys_df, ports_df, rng):
    """Aggregate journeys into route-level edges (mirrors the real edges schema)."""
    port_meta = ports_df.set_index('port_name')
    g = journeys_df.groupby(['origin_port', 'destination_port'])
    rows = []
    for (o, d), grp in g:
        if o == d:
            continue
        vt_counts = grp['ShiptypeLevel3'].value_counts()
        om, dm = port_meta.loc[o], port_meta.loc[d]
        rows.append({
            'origin_port': o,
            'destination_port': d,
            'total_journeys': int(len(grp)),
            'unique_vessels': int(grp['vessel_id'].nunique()),
            'mean_travel_time_hours': float(grp['travel_time_hours'].mean()),
            'median_travel_time_hours': float(grp['travel_time_hours'].median()),
            'std_travel_time_hours': float(grp['travel_time_hours'].std() or 0.0),
            'pct_international': float(om['country'] != dm['country']),
            'pct_intercontinental': float(om['continent'] != dm['continent']),
            'origin_country_first': om['country'],
            'destination_country_first': dm['country'],
            'origin_continent_first': om['continent'],
            'destination_continent_first': dm['continent'],
            'origin_sub_region_first': om['sub_region'],
            'destination_sub_region_first': dm['sub_region'],
            'mean_deadweight': float(rng.uniform(100, 30000)),
            'std_deadweight': float(rng.uniform(0, 10000)),
            'mean_draught_departure': float(grp['draught_departure'].mean()),
            'mean_draught_arrival': float(grp['draught_arrival'].mean()),
            'mean_vessel_age': float(rng.uniform(5, 30)),
            'dominant_vessel_type': vt_counts.idxmax(),
            'vessel_type_entropy': entropy(vt_counts.to_numpy(dtype=float)),
            'num_vessel_types': int(len(vt_counts)),
        })
    return pd.DataFrame(rows)


def make_nodes(ports_df, journeys_df, edges_df, rng):
    """Per-port aggregates on top of the geographic base (mirrors nodes schema)."""
    n = len(ports_df)
    dep = journeys_df.groupby('origin_port')
    arr = journeys_df.groupby('destination_port')
    out_deg = edges_df.groupby('origin_port').size()
    in_deg = edges_df.groupby('destination_port').size()

    def per_port(series, port, default=0.0):
        return float(series.get(port, default))

    departures = dep.size()
    arrivals = arr.size()
    dep_vessels = dep['vessel_id'].nunique()
    dwell = journeys_df.groupby('origin_port')['travel_time_hours']  # proxy stats

    rows = []
    for _, p in ports_df.iterrows():
        name = p['port_name']
        n_dep = per_port(departures, name)
        n_arr = per_port(arrivals, name)
        od = per_port(out_deg, name)
        idg = per_port(in_deg, name)
        vt = journeys_df[journeys_df['origin_port'] == name]['ShiptypeLevel3'].value_counts()
        rows.append({
            'port_name': name,
            'total_visits': int(n_dep + n_arr),
            'unique_vessels': int(per_port(dep_vessels, name)),
            'total_arrivals': int(n_arr),
            'total_departures': int(n_dep),
            'mean_dwell_time_hours': float(rng.uniform(5, 100)),
            'median_dwell_time_hours': float(rng.uniform(3, 60)),
            'std_dwell_time_hours': float(rng.uniform(0, 80)),
            'mean_deadweight': float(rng.uniform(100, 30000)),
            'max_deadweight': float(rng.uniform(30000, 400000)),
            'min_deadweight': float(rng.uniform(0, 100)),
            'mean_draught': float(rng.uniform(1, 10)),
            'max_draught': float(rng.uniform(10, 25)),
            'mean_vessel_age': float(rng.uniform(5, 30)),
            'dominant_vessel_type': vt.idxmax() if len(vt) else VESSEL_TYPES[0],
            'vessel_type_entropy': entropy(vt.to_numpy(dtype=float)) if len(vt) else 0.0,
            'latitude': p['latitude'],
            'longitude': p['longitude'],
            'country': p['country'],
            'country_name': p['country_name'],
            'continent': p['continent'],
            'sub_region': p['sub_region'],
            'out_degree': int(od),
            'outgoing_journeys': int(n_dep),
            'in_degree': int(idg),
            'incoming_journeys': int(n_arr),
            'total_degree': int(od + idg),
            'net_flow': float(n_dep - n_arr),
            'hub_score': float((od + idg) * np.log1p(n_dep + n_arr)),
        })
    assert len(rows) == n
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('data/synthetic'))
    ap.add_argument('--n-ports', type=int, default=300)
    ap.add_argument('--n-vessels', type=int, default=250)
    ap.add_argument('--n-journeys', type=int, default=30000,
                    help='approximate total number of journeys')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    print(f'Generating synthetic maritime data (seed={args.seed})...')

    ports_df, region_idx = make_ports(args.n_ports, rng)
    prefs = make_preferred_destinations(args.n_ports, region_idx, rng)
    journeys_df = make_journeys(ports_df, prefs, args.n_vessels, args.n_journeys, rng)
    edges_df = make_edges(journeys_df, ports_df, rng)
    nodes_df = make_nodes(ports_df, journeys_df, edges_df, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(args.out / 'nodes.parquet')
    edges_df.to_parquet(args.out / 'edges.parquet')
    journeys_df.to_parquet(args.out / 'journeys.parquet')

    print(f'  ports:    {len(nodes_df):>7,}  -> {args.out / "nodes.parquet"}')
    print(f'  routes:   {len(edges_df):>7,}  -> {args.out / "edges.parquet"}')
    print(f'  journeys: {len(journeys_df):>7,}  -> {args.out / "journeys.parquet"}')
    print('\nRun the pipeline on it with, e.g.:')
    print('  python experiments/maritime/run_dgi_training.py '
          '--config configs/experiments/synthetic.yaml --seed 42')


if __name__ == '__main__':
    main()
