# Maritime data schema

The maritime dataset is derived from AIS vessel-tracking data licensed from
S&P Global (Sea-web) and **cannot be redistributed**; see the Data Availability
section of the README. This document describes the exact schema of the three
parquet files the pipeline expects under `data/maritime/`, so that (a) readers
can interpret the feature engineering in `src/preprocessing/maritime.py`, and
(b) license holders can reconstruct compatible inputs. A schema-faithful
synthetic sample with the same columns can be generated with
`python scripts/make_synthetic_data.py` (written to `data/synthetic/`).

## nodes.parquet — ports (4,325 rows x 29 columns)

One row per port with aggregated static features (28 features; `port_name` is
the identifier):

```
['port_name', 'total_visits', 'unique_vessels', 'total_arrivals',
 'total_departures', 'mean_dwell_time_hours', 'median_dwell_time_hours',
 'std_dwell_time_hours', 'mean_deadweight', 'max_deadweight',
 'min_deadweight', 'mean_draught', 'max_draught', 'mean_vessel_age',
 'dominant_vessel_type', 'vessel_type_entropy', 'latitude', 'longitude',
 'country', 'country_name', 'continent', 'sub_region', 'out_degree',
 'outgoing_journeys', 'in_degree', 'incoming_journeys', 'total_degree',
 'net_flow', 'hub_score']
```

Column groups (see `NODE_*_COLS` in `src/preprocessing/maritime.py` for the
exact preprocessing assignment):

- **Identifier**: `port_name`.
- **Traffic aggregates**: visit/arrival/departure counts, unique vessels,
  dwell-time statistics (hours).
- **Vessel-mix aggregates**: deadweight/draught/age statistics,
  `dominant_vessel_type` (categorical), `vessel_type_entropy`.
- **Geography**: `latitude`, `longitude` (also geohash-encoded during
  preprocessing), `country`, `country_name`, `continent`, `sub_region`
  (UN M49-style sub-region; used as the diagnostic classification label).
- **Connectivity**: in/out/total degree, journey counts per direction,
  `net_flow`, `hub_score` (removed by the `exclude_connectivity_features`
  ablation).

## edges.parquet — directed routes (270,272 rows x 23 columns)

One row per directed port pair (origin, destination) with aggregated route
features (21 features; the pair is the edge identifier):

```
['origin_port', 'destination_port', 'total_journeys', 'unique_vessels',
 'mean_travel_time_hours', 'median_travel_time_hours',
 'std_travel_time_hours', 'pct_international', 'pct_intercontinental',
 'origin_country_first', 'destination_country_first',
 'origin_continent_first', 'destination_continent_first',
 'origin_sub_region_first', 'destination_sub_region_first',
 'mean_deadweight', 'std_deadweight', 'mean_draught_departure',
 'mean_draught_arrival', 'mean_vessel_age', 'dominant_vessel_type',
 'vessel_type_entropy', 'num_vessel_types']
```

- `total_journeys` is the full-period journey count for the route; it drives
  graph filtering (`min_journeys`), top-K sparsification, and the PortCity2Vec
  walk weights.
- `pct_international` / `pct_intercontinental` are binary indicators.
- The `*_first` columns are the origin/destination attributes as categoricals.

## journeys.parquet — individual vessel journeys (6.4M rows x 40 columns)

One row per completed port-to-port journey:

```
['journey_id', 'vessel_id', 'origin_port', 'destination_port',
 'departure_time', 'arrival_time', 'origin_lat', 'origin_lon',
 'destination_lat', 'destination_lon', 'origin_country',
 'destination_country', 'draught_departure', 'draught_arrival',
 'departure_port_visit_id', 'arrival_port_visit_id',
 'departure_movement_type', 'arrival_movement_type',
 'is_complete_journey', 'travel_time_hours', 'departure_year',
 'departure_month', 'departure_day_of_week', 'departure_hour',
 'origin_country_name', 'origin_continent', 'origin_sub_region',
 'destination_country_name', 'destination_continent',
 'destination_sub_region', 'Deadweight', 'FlagName', 'GrossTonnage',
 'YearOfBuild', 'ShiptypeLevel2', 'ShiptypeLevel3', 'ShiptypeLevel4',
 'ShiptypeLevel5', 'ShiptypeLevel5SubGroup', 'ShiptypeLevel5SubGroupType']
```

Journeys are only needed for downstream evaluation: warm-port computation,
temporal train/validation/test transition splits (`departure_time`), and the
per-vessel-type breakdown (`vessel_id` → `ShiptypeLevel3`). Journeys span
February 2022 – August 2023 and cover 80,639 unique vessels.

The loader (`src/data/maritime.py`) consumes only a subset of these columns:
`vessel_id`, `origin_port`, `destination_port`, `departure_time`, and
`ShiptypeLevel3`; the synthetic generator therefore emits that subset plus the
core journey attributes.

## Vessel types

`ShiptypeLevel3` values present in the dataset (ship-register taxonomy):

```
['Passenger/Ro-Ro Cargo', 'Passenger', 'Inland Waterways Tanker',
 'Towing/Pushing', 'Oil', 'Offshore Supply', 'Container',
 'Chemical', 'Liquefied Gas', 'Ro-Ro Cargo', 'General Cargo',
 'Bulk Dry', 'Inland Waterways Dry Cargo/Passenger', 'Unknown',
 'Other Bulk Dry', 'Fish Catching', 'Other Offshore',
 'Other Fishing', 'Dredging', 'Non Merchant', 'Other Dry Cargo',
 'Self Discharging Bulk Dry', 'Other Activities cont',
 'Passenger/General Cargo', 'Refrigerated Cargo',
 'Other Activities', 'Inland Waterways Other Non Seagoing',
 'Other Liquids', 'Research', 'Non Ship Structures',
 'Bulk Dry/Liquid', 'Non Propelled']
```

Labels are reproduced verbatim from the register taxonomy, including the
literal label `'Other Activities cont'`.

## Processed cache

On first load, `MaritimeDataset` filters the graph and writes a cache to
`<data_dir>/processed/` (`adj.npz`, `node_features.npz`, `edge_features.npy`,
`edge_index.npy`, `labels.npy`, `activity_labels.npy`, `splits.npz`,
`{train,val,test}_transitions.npy`, `vessel_type_map.json`, `metadata.json`).
The cache is derived from the licensed data and is likewise not distributed;
it regenerates automatically from the parquet files.
