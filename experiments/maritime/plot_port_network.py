"""Generate the global port-network figure for paper §3.3.1.

Combines the filtered graph (3,500 ports, 121,551 directed edges) with a
basemap world projection. Ports are drawn as scatter points sized and coloured
by total visits (log scale); the top-N most travelled edges are overlaid as
thin great-circle arcs.

Usage:
    python experiments/maritime/plot_port_network.py
    python experiments/maritime/plot_port_network.py --top-edges 5000 --out fig.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


def _gc_segments(lon1, lat1, lon2, lat2, npts=24):
    """Return a list of (lon_arr, lat_arr) for great-circle segments,
    split at antimeridian crossings so plotting doesn't draw a wrap-around line."""
    pts = _GEOD.npts(lon1, lat1, lon2, lat2, npts)
    lons = np.array([lon1] + [p[0] for p in pts] + [lon2])
    lats = np.array([lat1] + [p[1] for p in pts] + [lat2])

    # Split at antimeridian crossings (consecutive points jumping > 180 deg in lon)
    segs = []
    start = 0
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180:
            segs.append((lons[start:i], lats[start:i]))
            start = i
    segs.append((lons[start:], lats[start:]))
    return segs


def _load_filtered_graph(data_dir: Path):
    """Return nodes_df (3,500 ports w/ lat/lon/total_visits) and edges_df
    (filtered, with origin/destination port names and total_journeys)."""
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    edges = pd.read_parquet(data_dir / "edges.parquet")

    meta = json.loads((data_dir / "processed" / "metadata.json").read_text())
    port_to_idx = meta["port_to_idx"]
    kept_ports = set(port_to_idx.keys())

    nodes = nodes[nodes["port_name"].isin(kept_ports)].copy()
    nodes = nodes.dropna(subset=["latitude", "longitude"])

    edges = edges[
        edges["origin_port"].isin(kept_ports)
        & edges["destination_port"].isin(kept_ports)
        & (edges["origin_port"] != edges["destination_port"])
        & (edges["total_journeys"] >= 3)
    ].copy()

    edge_index = np.load(data_dir / "processed" / "edge_index.npy")
    idx_to_port = {v: k for k, v in port_to_idx.items()}
    filtered_pairs = {
        (idx_to_port[int(s)], idx_to_port[int(t)])
        for s, t in zip(edge_index[0], edge_index[1])
    }
    keep = [
        (o, d) in filtered_pairs
        for o, d in zip(edges["origin_port"], edges["destination_port"])
    ]
    edges = edges[keep].copy()

    print(f"  nodes (with lat/lon): {len(nodes):,}")
    print(f"  edges (filtered):     {len(edges):,}")
    return nodes, edges


def _annotate_hubs(m, nodes_df, ax, hubs):
    """Label hub ports. `hubs` is an iterable of (name, dx, dy) where the
    offset is in points; this lets us pull tightly clustered labels apart."""
    for name, dx, dy in hubs:
        row = nodes_df[nodes_df["port_name"].str.lower() == name.lower()]
        if row.empty:
            continue
        lat, lon = float(row.iloc[0]["latitude"]), float(row.iloc[0]["longitude"])
        x, y = m(lon, lat)
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="grey",
                alpha=0.88,
                linewidth=0.4,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="grey",
                linewidth=0.4,
                shrinkA=0,
                shrinkB=2,
            ),
            zorder=10,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/maritime"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/maritime_network.pdf"))
    parser.add_argument("--top-edges", type=int, default=0,
                        help="If >0, only the top-N scored edges are drawn. Default 0 = draw all edges.")
    parser.add_argument("--min-edge-km", type=float, default=0.0,
                        help="If >0, drop edges shorter than this great-circle distance in km. Default 0 = no filter.")
    parser.add_argument("--projection", type=str, default="robin",
                        help="Basemap projection (default: robin)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading filtered graph...")
    nodes, edges = _load_filtered_graph(args.data_dir)

    print("Building basemap...")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    m = Basemap(
        projection=args.projection,
        lon_0=0,
        resolution="l",
        ax=ax,
    )
    # Backgrounds: darker ocean (Image #1 style) lets cream edges stand out
    m.drawmapboundary(fill_color="#88aecc", linewidth=0.4)
    m.fillcontinents(color="#d5d5d5", lake_color="#88aecc")
    m.drawcoastlines(color="#525252", linewidth=0.35)
    m.drawcountries(color="#838383", linewidth=0.18)

    # ---- edges (all by default, drawn via LineCollection for speed) ----
    coord = nodes.set_index("port_name")[["latitude", "longitude"]].to_dict("index")

    # Optional distance filter (default off: --min-edge-km 0)
    olon = edges["origin_port"].map(lambda p: coord.get(p, {}).get("longitude"))
    olat = edges["origin_port"].map(lambda p: coord.get(p, {}).get("latitude"))
    dlon = edges["destination_port"].map(lambda p: coord.get(p, {}).get("longitude"))
    dlat = edges["destination_port"].map(lambda p: coord.get(p, {}).get("latitude"))
    valid = olon.notna() & olat.notna() & dlon.notna() & dlat.notna()
    edges = edges.loc[valid].copy()
    if args.min_edge_km > 0:
        _, _, dist_m = _GEOD.inv(
            olon[valid].values, olat[valid].values,
            dlon[valid].values, dlat[valid].values,
        )
        edges["distance_km"] = dist_m / 1000.0
        pre = len(edges)
        edges = edges[edges["distance_km"] >= args.min_edge_km].copy()
        print(f"  edges >= {args.min_edge_km} km: {len(edges):,}/{pre:,}")

    # Optional top-N selection (default off: --top-edges 0)
    if args.top_edges > 0:
        if "distance_km" in edges.columns:
            edges["score"] = np.log10(edges["total_journeys"]) + 0.4 * np.log10(edges["distance_km"])
        else:
            edges["score"] = np.log10(edges["total_journeys"])
        edges = edges.sort_values("score", ascending=False).head(args.top_edges).copy()

    print(f"Drawing all {len(edges):,} edges via LineCollection...")
    journey_max = float(edges["total_journeys"].max())
    journey_min = float(edges["total_journeys"].min())
    log_max = np.log10(journey_max)
    log_min = np.log10(journey_min)

    # Build segments (xy point arrays) and per-segment linewidth/alpha.
    # Per-edge alpha is kept low because 121K overlapping arcs would
    # otherwise saturate the figure; high-volume edges accumulate to a
    # visible cream colour, low-volume edges fill in a faint background.
    segments = []
    linewidths = []
    alphas = []
    for _, e in edges.iterrows():
        o = coord.get(e["origin_port"])
        d = coord.get(e["destination_port"])
        if o is None or d is None:
            continue
        lon1, lat1 = o["longitude"], o["latitude"]
        lon2, lat2 = d["longitude"], d["latitude"]
        if any(np.isnan([lon1, lat1, lon2, lat2])):
            continue
        # Skip near-antipodal pairs (npts can wrap oddly)
        if abs(lon1 - lon2) > 175 and abs(lat1 - lat2) < 5:
            continue

        w = (np.log10(e["total_journeys"]) - log_min) / max(log_max - log_min, 1e-9)
        try:
            for seg_lons, seg_lats in _gc_segments(lon1, lat1, lon2, lat2, npts=16):
                if len(seg_lons) < 2:
                    continue
                x_seg, y_seg = m(seg_lons, seg_lats)
                segments.append(np.column_stack([x_seg, y_seg]))
                linewidths.append(0.20 + 0.65 * w)
                alphas.append(0.04 + 0.30 * w)
        except (ValueError, ZeroDivisionError):
            continue

    # Build an (N, 4) RGBA array: constant cream RGB, per-segment alpha
    base_rgb = (1.00, 0.94, 0.70)  # #fff0b3 in 0-1 floats
    rgba = np.empty((len(alphas), 4), dtype=np.float64)
    rgba[:, 0] = base_rgb[0]
    rgba[:, 1] = base_rgb[1]
    rgba[:, 2] = base_rgb[2]
    rgba[:, 3] = alphas

    lc = LineCollection(
        segments,
        colors=rgba,
        linewidths=linewidths,
        zorder=2,
        rasterized=True,
        capstyle="round",
    )
    ax.add_collection(lc)

    # ---- ports (all 3,500, coloured by log10 visits) ----
    print("Drawing ports...")
    visits = nodes["total_visits"].clip(lower=1).values
    x, y = m(nodes["longitude"].values, nodes["latitude"].values)
    size = 1.5 + 10.0 * (np.log10(visits) / np.log10(visits.max()))
    # rasterized=True embeds the scatter as a bitmap inside the PDF; without
    # it small vector dots can lose colour separation when the PDF viewer
    # anti-aliases. PNG previews are already raster and unaffected.
    sc = ax.scatter(
        x, y,
        c=visits,
        cmap="OrRd",
        norm=LogNorm(vmin=max(visits.min(), 1), vmax=visits.max()),
        s=size,
        edgecolors="#1f1f1f",
        linewidths=0.18,
        zorder=5,
        rasterized=True,
    )

    # ---- label major hubs (name, dx, dy) where (dx,dy) is offset in points ----
    hubs = [
        ("Rotterdam",   18, 14),
        ("Hamburg",     22, -14),
        ("Algeciras",  -12, -18),
        ("Suez",         10,  10),
        ("Singapore",   -8, -16),
        ("Port Klang",  18,  12),
        ("Shanghai",    14,   8),
        ("Los Angeles", -55, -10),
        ("Houston",     -8, -16),
        ("Santos",      10, -12),
        ("Durban",      10, -10),
    ]
    _annotate_hubs(m, nodes, ax, hubs)

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.04, shrink=0.55, aspect=40)
    cbar.set_label("Total port visits (log scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Figure title
    ax.set_title("Global Maritime Port-Visit Network", fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    # Also a PNG companion for quick previews
    png_out = args.out.with_suffix(".png")
    fig.savefig(png_out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}  and  {png_out}")


if __name__ == "__main__":
    main()
