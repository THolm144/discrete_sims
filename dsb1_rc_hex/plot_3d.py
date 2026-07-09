"""
plot_3d.py
==========
Headless 3-D optical photon visualiser.  Renders hit scatter + geometry
wireframes from world-provided geometry primitives.

World get_geometry_primitives() contract
-----------------------------------------
    def get_geometry_primitives() -> list[dict]

    Each dict describes one volume:
        type     : "box" | "tube"           (required)
        center   : [cx, cy, cz]  cm         (required)
        color    : "#rrggbb"                 (required)
        label    : str                       (required)
        alpha    : float  0-1               (default 0.6)
        linewidth: float                    (default 0.8)

        box  extras:  half  : [hx, hy, hz]  cm
        tube extras:  rmax  : float cm
                      rmin  : float cm  (default 0, solid)
                      height: float cm

    If the world does not provide this hook, a simple bounding box of the
    target volume is drawn instead (using PHANTOM_CM).

Views rendered
--------------
    perspective  elev=20  azim=-60
    top          elev=90  azim=0    (beam's-eye)
    side_xz      elev=0   azim=0
    side_yz      elev=0   azim=90

Outputs (all in batch_dir):
    3d_perspective.png
    3d_top.png
    3d_side_xz.png
    3d_side_yz.png
    3d_panel.png   (2×2 composite)

Usage:
    python3 plot_3d.py
    python3 plot_3d.py --world scintx_sipm_array --max-tracks 500
    python3 plot_3d.py --batch-dir runs/scintx_sipm_array/500000keV_20250101_120000
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import analysis_utils as utils


# ─────────────────────────────────────────────────────────────────────────────
# VIEW DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

VIEW_PARAMS: dict[str, tuple[int, int, str]] = {
    "perspective": (20, -60, "Perspective"),
    "top":         (90,   0, "Top (beam's-eye)"),
    "side_xz":     ( 0,   0, "Side XZ"),
    "side_yz":     ( 0,  90, "Side YZ"),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="3-D optical photon visualiser (headless)")
    p.add_argument("--batch-dir",  default=None)
    p.add_argument("--world",      default=None)
    p.add_argument("--max-tracks", type=int, default=300)
    p.add_argument("--dpi",        type=int, default=150)
    p.add_argument("--views", nargs="+", default=list(VIEW_PARAMS.keys()),
                   choices=list(VIEW_PARAMS.keys()))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# WORLD LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_world(world_name: str, script_dir: Path):
    sys.path.insert(0, str(script_dir / "worlds"))
    try:
        return importlib.import_module(world_name)
    except ModuleNotFoundError:
        return None


def load_run_metadata(run_dir: Path) -> dict:
    path = run_dir / "sim_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY PRIMITIVES → EDGE LISTS
# ─────────────────────────────────────────────────────────────────────────────

def _box_edges(cx: float, cy: float, cz: float,
               hx: float, hy: float, hz: float) -> list[np.ndarray]:
    """Return list of (2,3) arrays — one per edge — for Line3DCollection."""
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    corners = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])
    pairs = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return [np.array([corners[a], corners[b]]) for a, b in pairs]


def _tube_edges(cx: float, cy: float, cz: float,
                rmax: float, height: float,
                rmin: float = 0.0, n_seg: int = 24) -> list[np.ndarray]:
    """Approximate tube/cylinder wireframe with n_seg-gon top and bottom rings."""
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    edges  = []
    hz     = height / 2.0

    for r in ([rmax] if rmin == 0.0 else [rmin, rmax]):
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        for sign in (-1, 1):
            zz = cz + sign * hz
            ring = np.column_stack([xs, ys, np.full(n_seg, zz)])
            for i in range(n_seg):
                edges.append(np.array([ring[i], ring[(i+1) % n_seg]]))

        # Vertical lines (outer only, or both for annuli)
        xs_top = cx + r * np.cos(angles[::4])
        ys_top = cy + r * np.sin(angles[::4])
        for x_, y_ in zip(xs_top, ys_top):
            edges.append(np.array([[x_, y_, cz - hz], [x_, y_, cz + hz]]))

    return edges


def _prim_to_edges(prim: dict) -> list[np.ndarray]:
    cx, cy, cz = prim["center"]
    t = prim.get("type", "box")
    if t == "box":
        hx, hy, hz = prim["half"]
        return _box_edges(cx, cy, cz, hx, hy, hz)
    elif t == "tube":
        return _tube_edges(
            cx, cy, cz,
            prim["rmax"], prim["height"],
            rmin=prim.get("rmin", 0.0),
        )
    else:
        return []


def build_geometry_collections(primitives: list[dict]
                                ) -> list[tuple[list, str, str, float, float]]:
    """
    Convert primitive dicts to (edges, color, label, linewidth, alpha) tuples
    ready for Line3DCollection.  Duplicate labels are suppressed after the first.
    """
    seen   = set()
    result = []
    for prim in primitives:
        label = prim.get("label", "")
        lbl   = label if label not in seen else ""
        seen.add(label)
        result.append((
            _prim_to_edges(prim),
            prim.get("color", "#00ffcc"),
            lbl,
            prim.get("linewidth", 0.8),
            prim.get("alpha", 0.6),
        ))
    return result


def default_geometry_primitives(phantom_cm: list) -> list[dict]:
    """Fallback: single bounding box for the target volume."""
    return [{
        "type":      "box",
        "center":    [0.0, 0.0, 0.0],
        "half":      [phantom_cm[0]/2, phantom_cm[1]/2, phantom_cm[2]/2],
        "color":     "#00ffcc",
        "label":     "Target",
        "alpha":     0.4,
        "linewidth": 1.0,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# BEAM ARROW
# ─────────────────────────────────────────────────────────────────────────────

def _beam_arrow_endpoints(beam_cfg: dict, phantom_cm: list
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (source_pos, target_pos) in cm for the beam trajectory line.
    Uses beam_config from metadata if present; falls back to -Z convention.
    """
    if beam_cfg:
        direction  = np.array(beam_cfg.get("direction",  [0, 0, 1]), dtype=float)
        target_cm  = np.array(beam_cfg.get("target_cm",  [0, 0, 0]), dtype=float)
        offset_cm  = beam_cfg.get("offset_cm", 2.0)
        direction /= np.linalg.norm(direction)
        source = target_cm - direction * offset_cm
        # Show arrow ending just outside the entry face
        entry  = target_cm - direction * (phantom_cm[2] / 2 + 0.1)
        return source, entry
    else:
        hz     = phantom_cm[2] / 2
        source = np.array([0, 0, -(hz + 2.0)])
        entry  = np.array([0, 0, -(hz + 0.1)])
        return source, entry


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_view(ax, hit_x, hit_y, hit_z,
                exit_x, exit_y, exit_z,
                geom_collections: list,
                phantom_cm: list,
                beam_cfg: dict,
                elev: int, azim: int):

    ax.set_facecolor("black")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#222222")
    ax.tick_params(colors="white", labelsize=6)

    # Scatter: detector hits
    if len(hit_x):
        ax.scatter(hit_x, hit_y, hit_z,
                   c="#ff007f", s=1.5, alpha=0.7, depthshade=False,
                   label=f"Detector hits ({len(hit_x)})")

    # Scatter: optical exits (missed detector)
    if len(exit_x):
        ax.scatter(exit_x, exit_y, exit_z,
                   c="#00cfff", s=1.0, alpha=0.4, depthshade=False,
                   label=f"Volume exits ({len(exit_x)})")

   # Geometry wireframes
    for edges, color, label, lw, alpha in geom_collections:
        lbl = label if label else None
        try:
            ax.add_collection3d(
                Line3DCollection(edges, colors=color, linewidths=lw, alpha=alpha, label=lbl)
            )
        except TypeError:
            # Safely bypass older matplotlib 3D internal auto-scaling bugs
            ax.add_collection(
                Line3DCollection(edges, colors=color, linewidths=lw, alpha=alpha, label=lbl)
            )

    # Beam arrow
    src, entry = _beam_arrow_endpoints(beam_cfg, phantom_cm)
    ax.plot([src[0], entry[0]], [src[1], entry[1]], [src[2], entry[2]],
            color="yellow", linewidth=1.5, alpha=0.9, label="Beam")
    ax.scatter(*src, color="yellow", s=20, zorder=5)

    # Axis limits
    pad   = 0.5
    lim_r = max(phantom_cm[0], phantom_cm[1]) / 2 + pad
    all_z = list(hit_z) + list(exit_z) + [src[2], entry[2]]
    lim_z = max(abs(float(np.min(all_z) if all_z else 0)),
                abs(float(np.max(all_z) if all_z else 1)),
                phantom_cm[2] / 2) + pad

    ax.set_xlim(-lim_r, lim_r)
    ax.set_ylim(-lim_r, lim_r)
    ax.set_zlim(-lim_z, lim_z)

    ax.set_xlabel("X (cm)", color="white", fontsize=7, labelpad=2)
    ax.set_ylabel("Y (cm)", color="white", fontsize=7, labelpad=2)
    ax.set_zlabel("Z (cm)", color="white", fontsize=7, labelpad=2)
    ax.view_init(elev=elev, azim=azim)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    batch_dir = utils.find_batch_dir(script_dir, args.world, args.batch_dir)
    run_dirs  = utils.find_runs(batch_dir)
    run_dir   = run_dirs[0]   # use run_0 for track data

    meta        = load_run_metadata(run_dir)
    world_name  = args.world or meta.get("world", "unknown")
    world       = load_world(world_name, script_dir)
    phantom_cm  = (world.PHANTOM_CM if world
                   else meta.get("phantom_cm", [10, 10, 0.6]))
    beam_cfg    = meta.get("beam_config", {})

    print(f"  Batch dir   : {batch_dir}")
    print(f"  World       : {world_name}")
    print(f"  Phantom     : {phantom_cm}")

    # ── Geometry primitives ───────────────────────────────────────────────
    if world and hasattr(world, "get_geometry_primitives"):
        primitives = world.get_geometry_primitives()
        print(f"  Geometry    : {len(primitives)} primitives from world hook")
    else:
        primitives = default_geometry_primitives(phantom_cm)
        print("  Geometry    : fallback bounding box (no world hook)")

    geom_collections = build_geometry_collections(primitives)

    # ── Track data ────────────────────────────────────────────────────────
    half = args.max_tracks // 2
    hit_files  = list(run_dir.glob("detector_hits*.root"))
    exit_files = [run_dir / "optical_exited.root"]

    hit_x,  hit_y,  hit_z  = utils.load_root_positions(hit_files,  half)
    exit_x, exit_y, exit_z = utils.load_root_positions(exit_files, half)
    print(f"  Loaded      : {len(hit_x)} hits, {len(exit_x)} exits")

    # ── Render ────────────────────────────────────────────────────────────
    out_paths = []

    for view_name in args.views:
        elev, azim, title = VIEW_PARAMS[view_name]
        print(f"  Rendering   : {view_name}")

        fig = plt.figure(figsize=(10, 8), facecolor="black")
        ax  = fig.add_subplot(111, projection="3d")
        render_view(ax, hit_x, hit_y, hit_z, exit_x, exit_y, exit_z,
                    geom_collections, phantom_cm, beam_cfg, elev, azim)
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#111111", edgecolor="#444444", labelcolor="white")
        fig.suptitle(f"{world_name}  |  {title}", color="white", fontsize=9, y=0.98)

        out = batch_dir / f"3d_{view_name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        print(f"    → {out.name}")
        out_paths.append(out)

    # ── 2×2 panel ────────────────────────────────────────────────────────
    views_to_use = args.views[:4]
    n_cols = 2
    n_rows = (len(views_to_use) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(16, 12), facecolor="black",
                             subplot_kw={"projection": "3d"})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, vname in zip(axes_list, views_to_use):
        elev, azim, title = VIEW_PARAMS[vname]
        render_view(ax, hit_x, hit_y, hit_z, exit_x, exit_y, exit_z,
                    geom_collections, phantom_cm, beam_cfg, elev, azim)
        ax.set_title(title, color="white", fontsize=8, pad=2)

    # Hide any unused subplot slots
    for ax in axes_list[len(views_to_use):]:
        ax.set_visible(False)

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               facecolor="#111111", edgecolor="#444444", labelcolor="white")

    energy_mev = meta.get("energy_kev", 0) / 1000
    n_prim     = meta.get("n_primaries", "?")
    fig.suptitle(
        f"{world_name}  —  {energy_mev:.0f} MeV  |  N={n_prim}  |  "
        f"hits: {len(hit_x)}  exits: {len(exit_x)}",
        color="white", fontsize=10, y=1.005,
    )

    panel_out = batch_dir / "3d_panel.png"
    fig.savefig(panel_out, dpi=args.dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    print(f"  → {panel_out.name}")
    out_paths.append(panel_out)

    print(f"\nDone. {len(out_paths)} images saved to {batch_dir}")


if __name__ == "__main__":
    main()