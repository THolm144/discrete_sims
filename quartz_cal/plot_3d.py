"""
plot_3d.py (Fixed for Array Repetition and Track Key Identification)
==========
Headless 3-D optical photon visualiser.  Renders hit scatter + geometry
wireframes from world-provided geometry primitives. Now replicates primitives
matching grid layout allocations and loads tracks correctly.
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
from matplotlib.lines import Line2D
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

TRACK_STYLES = {
    "Cerenkov":      ("#1138e2", "Cerenkov"),
    "Scintillation": ("#ff9900", "Scintillation"),
    "other":         ("#aaaaaa", "Other"),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="3-D optical photon visualiser (headless)")
    p.add_argument("--batch-dir",  default=None)
    p.add_argument("--world",      default=None)
    p.add_argument("--max-tracks", type=int, default=300)
    p.add_argument("--max-optical-steps", type=int, default=5000)
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

def _box_edges(cx, cy, cz, hx, hy, hz) -> list[np.ndarray]:
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


def _tube_edges(cx, cy, cz, rmax, height, rmin=0.0, n_seg=24) -> list[np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    edges  = []
    hz     = height / 2.0
    for r in ([rmax] if rmin == 0.0 else [rmin, rmax]):
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)
        for sign in (-1, 1):
            zz   = cz + sign * hz
            ring = np.column_stack([xs, ys, np.full(n_seg, zz)])
            for i in range(n_seg):
                edges.append(np.array([ring[i], ring[(i+1) % n_seg]]))
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
        return _tube_edges(cx, cy, cz, prim["rmax"], prim["height"],
                           rmin=prim.get("rmin", 0.0))
    return []


def build_geometry_collections(primitives, world_mod=None):
    """Parses layout primitives and automatically generates NxN grids for array elements"""
    seen   = set()
    result = []
    
    # Check if we can determine the grid size N from the world module context
    N = getattr(world_mod, "DEFAULT_N", 10) if world_mod else 10
    pitch_cm = 0.5  # 5.0 mm converted to cm
    
    for prim in primitives:
        label = prim.get("label", "")
        
        # If the primitive belongs to an individual component slice, tile it across the array layout
        if "pixel" in label.lower() or "sipm" in label.lower():
            lbl_registered = False
            start_offset = -(N - 1) * pitch_cm / 2.0
            
            for xi in range(N):
                for yi in range(N):
                    dx = start_offset + xi * pitch_cm
                    dy = start_offset + yi * pitch_cm
                    
                    copied_prim = prim.copy()
                    cx, cy, cz = copied_prim["center"]
                    copied_prim["center"] = [cx + dx, cy + dy, cz]
                    
                    lbl = label if (label not in seen and not lbl_registered) else ""
                    if lbl: 
                        seen.add(label)
                        lbl_registered = True
                        
                    result.append((
                        _prim_to_edges(copied_prim),
                        prim.get("color", "#ff00c8"),
                        lbl,
                        prim.get("linewidth", 0.8),
                        prim.get("alpha", 0.6),
                    ))
        else:
            lbl = label if label not in seen else ""
            seen.add(label)
            result.append((
                _prim_to_edges(prim),
                prim.get("color", "#ff00c8"),
                lbl,
                prim.get("linewidth", 0.8),
                prim.get("alpha", 0.6),
            ))
            
    return result


def default_geometry_primitives(phantom_cm):
    return [{
        "type":      "box",
        "center":    [0.0, 0.0, 0.0],
        "half":      [phantom_cm[0]/2, phantom_cm[1]/2, phantom_cm[2]/2],
        "color":     "#ff006a",
        "label":     "Target",
        "alpha":     0.4,
        "linewidth": 1.0,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL TRACK LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_optical_tracks(track_file: Path, max_steps: int) -> dict[str, list[np.ndarray]]:
    empty = {"Cerenkov": [], "Scintillation": [], "other": []}
    if not track_file.exists():
        return empty

    try:
        import uproot
    except ModuleNotFoundError:
        print("  WARNING: 'uproot' package missing. Cannot parse optical track steps.")
        return empty

    with uproot.open(track_file) as f:
        tree_keys = [k for k in f.keys() if "optical_tracks" in k or "optical_tracker" in k]
        if not tree_keys:
            return empty

        tree = f[tree_keys[0]]
        all_branch_names = [k for k in tree.keys()]

        # Find an available time column
        time_col = None
        for candidate in ["GlobalTime", "TrackTime", "Time", "Time_ns"]:
            if candidate in all_branch_names:
                time_col = candidate
                break

        # Match position keys
        x_col = "Position_X" if "Position_X" in all_branch_names else "PostPosition_X"
        y_col = "Position_Y" if "Position_Y" in all_branch_names else "PostPosition_Y"
        z_col = "Position_Z" if "Position_Z" in all_branch_names else "PostPosition_Z"

        # Check if TrackCreatorProcess exists
        has_process = "TrackCreatorProcess" in all_branch_names

        keys = ["EventID", "TrackID", x_col, y_col, z_col]
        if time_col:
            keys.append(time_col)
        if has_process:
            keys.append("TrackCreatorProcess")

        data = tree.arrays(keys, library="np")

        if len(data["EventID"]) == 0:
            return empty

        n_steps = min(len(data["EventID"]), max_steps)
        ev_ids  = data["EventID"][:n_steps]
        tr_ids  = data["TrackID"][:n_steps]
        x_cm    = data[x_col][:n_steps] / 10.0
        y_cm    = data[y_col][:n_steps] / 10.0
        z_cm    = data[z_col][:n_steps] / 10.0
        
        times   = data[time_col][:n_steps] if time_col else np.arange(n_steps)
        procs   = data["TrackCreatorProcess"][:n_steps] if has_process else None

        uid = (ev_ids.astype(np.int64) * 10000000) + tr_ids.astype(np.int64)

        # Map track process if available
        track_process = {}
        for track_key in np.unique(uid):
            if has_process and procs is not None:
                mask = (uid == track_key)
                first_proc = procs[mask][0]
                if isinstance(first_proc, bytes):
                    first_proc = first_proc.decode()
                track_process[track_key] = first_proc
            else:
                track_process[track_key] = "unknown"

        result = {"Cerenkov": [], "Scintillation": [], "other": []}

        for track_key in np.unique(uid):
            mask = (uid == track_key)
            if np.sum(mask) < 2:
                continue
            t_sort = np.argsort(times[mask])
            seg = np.column_stack([x_cm[mask][t_sort],
                                   y_cm[mask][t_sort],
                                   z_cm[mask][t_sort]])
            
            proc = track_process[track_key]
            if "Cerenkov" in proc or "Cherenkov" in proc:
                result["Cerenkov"].append(seg)
            elif "Scintillation" in proc:
                result["Scintillation"].append(seg)
            else:
                result["other"].append(seg)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# BEAM ARROW
# ─────────────────────────────────────────────────────────────────────────────

def _beam_arrow_endpoints(beam_cfg, phantom_cm):
    if beam_cfg:
        direction = np.array(beam_cfg.get("direction", [0, 0, 1]), dtype=float)
        target_cm = np.array(beam_cfg.get("target_cm", [0, 0, 0]), dtype=float)
        offset_cm = beam_cfg.get("offset_cm", 2.0)
        direction /= np.linalg.norm(direction)
        source = target_cm - direction * offset_cm
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
                optical_lines: dict,
                geom_collections: list,
                phantom_cm: list,
                beam_cfg: dict,
                elev: int, azim: int):

    ax.set_facecolor("black")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#222222")
    ax.tick_params(colors="white", labelsize=6)

    proxy_handles = []

    if len(hit_x):
        ax.scatter(hit_x, hit_y, hit_z,
                   c="#ff007f", s=1.5, alpha=0.7, depthshade=False,
                   label=f"Detector hits ({len(hit_x)})")
        proxy_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff007f",
                   markersize=5, label=f"Detector hits ({len(hit_x)})", linestyle="None")
        )

    if len(exit_x):
        ax.scatter(exit_x, exit_y, exit_z,
                   c="#00cfff", s=1.0, alpha=0.4, depthshade=False,
                   label=f"Volume exits ({len(exit_x)})")
        proxy_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#00cfff",
                   markersize=5, label=f"Volume exits ({len(exit_x)})", linestyle="None")
        )

    for proc_key, (color, label) in TRACK_STYLES.items():
        lines = optical_lines.get(proc_key, [])
        if not lines:
            continue
        coll = Line3DCollection(lines, colors=color, linewidths=0.5, alpha=0.25)
        ax.add_collection3d(coll, autolim=False)
        proxy_handles.append(
            Line2D([0], [0], color=color, linewidth=1.5,
                   label=f"{label} paths ({len(lines)})")
        )

    for edges, color, label, lw, alpha in geom_collections:
        ax.add_collection3d(
            Line3DCollection(edges, colors=color, linewidths=lw, alpha=alpha)
        )
        if label:
            proxy_handles.append(
                Line2D([0], [0], color=color, linewidth=lw, label=label)
            )

    src, entry = _beam_arrow_endpoints(beam_cfg, phantom_cm)
    ax.plot([src[0], entry[0]], [src[1], entry[1]], [src[2], entry[2]],
            color="yellow", linewidth=1.5, alpha=0.9)
    ax.scatter(*src, color="yellow", s=20, zorder=5)
    proxy_handles.append(
        Line2D([0], [0], color="yellow", linewidth=1.5, label="Beam")
    )

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

    return proxy_handles


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    batch_dir = utils.find_batch_dir(script_dir, args.world, args.batch_dir)
    run_dirs  = utils.find_runs(batch_dir)
    run_dir   = run_dirs[0]

    meta       = load_run_metadata(run_dir)
    world_name = args.world or meta.get("world", "unknown")
    world      = load_world(world_name, script_dir)
    phantom_cm = (world.PHANTOM_CM if world
                  else meta.get("phantom_cm", [10, 10, 0.6]))
    beam_cfg   = meta.get("beam_config", {})

    print(f"  Batch dir   : {batch_dir}")
    print(f"  World       : {world_name}")
    print(f"  Phantom     : {phantom_cm}")

    if world and hasattr(world, "get_geometry_primitives"):
        primitives = world.get_geometry_primitives()
        print(f"  Geometry    : {len(primitives)} primitive specifications loaded from world.")
    else:
        primitives = default_geometry_primitives(phantom_cm)
        print("  Geometry    : fallback bounding box (no world hook)")

    # FIXED: Pass world module structure to automatically loop arrays across NxN rows
    geom_collections = build_geometry_collections(primitives, world)

    half       = args.max_tracks // 2
    hit_files  = list(run_dir.glob("detector_hits*.root"))
    exit_files = [run_dir / "optical_exited.root"]

    hit_x,  hit_y,  hit_z  = utils.load_root_positions(hit_files,  half)
    exit_x, exit_y, exit_z = utils.load_root_positions(exit_files, half)
    print(f"  Loaded      : {len(hit_x)} hits, {len(exit_x)} exits")

    track_root_path = run_dir / "optical_tracks.root"
    optical_lines   = load_optical_tracks(track_root_path, args.max_optical_steps)
    n_cer   = len(optical_lines["Cerenkov"])
    n_scint = len(optical_lines["Scintillation"])
    n_other = len(optical_lines["other"])
    if n_cer + n_scint + n_other:
        print(f"  Loaded Track: {n_cer} Cerenkov, {n_scint} Scintillation, "
              f"{n_other} other segments")

    out_paths = []

    for view_name in args.views:
        elev, azim, title = VIEW_PARAMS[view_name]
        print(f"  Rendering   : {view_name}")

        fig = plt.figure(figsize=(10, 8), facecolor="black")
        ax  = fig.add_subplot(111, projection="3d")
        proxy_handles = render_view(
            ax, hit_x, hit_y, hit_z, exit_x, exit_y, exit_z,
            optical_lines, geom_collections, phantom_cm, beam_cfg, elev, azim
        )
        if proxy_handles:
            ax.legend(handles=proxy_handles, loc="upper left", fontsize=7,
                      facecolor="#111111", edgecolor="#444444", labelcolor="white")
        fig.suptitle(f"{world_name}  |  {title}", color="white", fontsize=9, y=0.98)

        out = batch_dir / f"3d_{view_name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        print(f"    → {out.name}")
        out_paths.append(out)

    views_to_use = args.views[:4]
    n_cols = 2
    n_rows = (len(views_to_use) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(16, 12), facecolor="black",
                             subplot_kw={"projection": "3d"})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    all_proxy_handles = None
    for ax, vname in zip(axes_list, views_to_use):
        elev, azim, title = VIEW_PARAMS[vname]
        proxy_handles = render_view(
            ax, hit_x, hit_y, hit_z, exit_x, exit_y, exit_z,
            optical_lines, geom_collections, phantom_cm, beam_cfg, elev, azim
        )
        ax.set_title(title, color="white", fontsize=8, pad=2)
        if all_proxy_handles is None:
            all_proxy_handles = proxy_handles

    for ax in axes_list[len(views_to_use):]:
        ax.set_visible(False)

    if all_proxy_handles:
        fig.legend(handles=all_proxy_handles, loc="lower center", ncol=4, fontsize=8,
                   facecolor="#111111", edgecolor="#444444", labelcolor="white")

    energy_mev = meta.get("energy_kev", 0) / 1000
    n_prim     = meta.get("n_primaries", "?")
    n_tracks   = n_cer + n_scint + n_other
    fig.suptitle(
        f"{world_name}  —  {energy_mev:.0f} MeV  |  N={n_prim}  |  "
        f"hits: {len(hit_x)}  exits: {len(exit_x)}  paths: {n_tracks}",
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