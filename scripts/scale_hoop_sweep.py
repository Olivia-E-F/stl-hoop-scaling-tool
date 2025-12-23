"""
Sweep-based hoop scaling.

Goal:
- Scale the hoop uniformly along its swept path
- Preserve circularity everywhere
- Keep the mount geometry unchanged

This script currently performs diagnostics only.
No geometry is modified yet.
"""

import struct
import numpy as np
from pathlib import Path


# ---------------------------------
# STL IO (read or write)
# ---------------------------------

def read_binary_stl(path):
    # Binary STL layout is fixed: header + triangle count + triangle records
    with open(path, "rb") as f:
        f.read(80)  # header is unused but must be consumed
        tri_count = struct.unpack("<I", f.read(4))[0]

        tri_dtype = np.dtype([
            ("normal", np.float32, (3,)),
            ("verts",  np.float32, (3, 3)),
            ("attr",   np.uint16)
        ])

        data = np.fromfile(f, dtype=tri_dtype, count=tri_count)

    # Flatten triangle vertices so each row is a single 3D point
    return data["verts"].reshape(-1, 3)

def write_binary_stl(path, header, triangles):
    with open(path, "wb") as f:
        # Header is exactly 80 bytes
        if len(header) < 80:
            header = header + b" " * (80 - len(header))
        f.write(header[:80])

        # Number of triangles
        f.write(struct.pack("<I", len(triangles)))

        # Write triangle data as is
        triangles.tofile(f)


# ---------------------------------
# Centerline extraction
# ---------------------------------

def extract_centerline(verts, x_bins=200):
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # Sweep is primarily along X, so we bin vertices by X position
    x_min, x_max = x.min(), x.max() # sweep interval
    bins = np.linspace(x_min, x_max, x_bins + 1) # to define boundaries for slices (edges)

    centers = []

    for i in range(x_bins):
        in_bin = (x >= bins[i]) & (x < bins[i + 1])
        if not np.any(in_bin):
            continue  # skip empty slices

        cx = 0.5 * (bins[i] + bins[i + 1])  # representative X for this slice
        cy = y[in_bin].mean()               # Y centroid approximates sweep center
        cz = z[in_bin].mean()               # Z centroid approximates sweep center

        centers.append([cx, cy, cz])

    return np.array(centers)


# ---------------------------------
# Local frames
# ---------------------------------

def compute_tangents(centerline):
    tangents = np.zeros_like(centerline)

    for i in range(len(centerline)):
        # Forward/backward differences at ends, central difference otherwise
        if i == 0:
            v = centerline[i + 1] - centerline[i]
        elif i == len(centerline) - 1:
            v = centerline[i] - centerline[i - 1]
        else:
            v = centerline[i + 1] - centerline[i - 1]

        tangents[i] = v / np.linalg.norm(v)  # normalize so projection math is stable

    return tangents


def local_radius(vertex, center, tangent):
    # Vector from centerline to vertex
    d = vertex - center

    # Remove component along the sweep direction
    d_parallel = np.dot(d, tangent) * tangent
    d_perp = d - d_parallel

    # Remaining magnitude is distance in the local cross-section plane
    return np.linalg.norm(d_perp)


# ---------------------------------
# Vertex association and scaling math (diagnostic)
# ---------------------------------

def associate_vertices_to_slices(verts, centerline):
    x_centers = centerline[:, 0]
    x_verts = verts[:, 0]

    # Nearest-slice association based on X is sufficient for a monotonic sweep
    idx = np.searchsorted(x_centers, x_verts) # idx[i]=s  vertex i uses s's center+tan
    return np.clip(idx, 0, len(x_centers) - 1) # no val negative or outside of slice


def compute_scaled_radii(verts, centerline, tangents, scale):
    slice_idx = associate_vertices_to_slices(verts, centerline)

    r_orig = np.zeros(len(verts))
    r_new = np.zeros(len(verts))

    # Measure local radius in the appropriate normal plane
    for i in range(len(verts)):
        s = slice_idx[i]
        r_orig[i] = local_radius(
            verts[i],
            centerline[s],
            tangents[s]
        )

    # Mean radius defines the local profile size for each slice
    r_mean = np.zeros(len(centerline))
    for s in range(len(centerline)):
        mask = slice_idx == s
        if np.any(mask):
            r_mean[s] = r_orig[mask].mean()

    # CAD-style scaling: preserve profile shape relative to its local mean
    ##   r_orig[i] = where the vertex currently sits
    ##   r_mean[s] = center of the profile at this slice
    ##   (r_orig[i] - r_mean[s]) = offset from profile center

    for i in range(len(verts)):
        s = slice_idx[i]
        r_new[i] = r_mean[s] + scale * (r_orig[i] - r_mean[s])

    return r_orig, r_new, r_mean

# ---------------------------------
# Apply scaling to vertices
# ---------------------------------

def apply_scaled_vertices(verts, centerline, tangents, r_orig, r_new):
    slice_idx = associate_vertices_to_slices(verts, centerline)
    # each vertex has a local frame defined by centerline slice

    verts_new = verts.copy() # for later comarisons 

    for i in range(len(verts)):
        s = slice_idx[i]

        center = centerline[s]
        tangent = tangents[s]

        d = verts[i] - center

        # split into sweep and cross-section components
        # projection of d onto tangent: d_parallel = (d · T_s) T_s
        d_parallel = np.dot(d, tangent) * tangent
        # perpendicular component in normal plane: d_perp = d - d_parallel
        d_perp = d - d_parallel

        # scale factor to map |d_perp| = r_orig -> r_new
        if r_orig[i] > 0:
            scale_factor = r_new[i] / r_orig[i]
        else:
            scale_factor = 1.0

        # scaled perpendicular component: d_perp' = scale * d_perp
        d_perp_new = d_perp * scale_factor
        # reconstructed vertex: V' = C_s + d_parallel + d_perp'
        verts_new[i] = center + d_parallel + d_perp_new
        
    return verts_new


# ---------------------------------
# Main (diagnostics only)
# ---------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    stl_path = repo_root / "data" / "original" / "Sports_Ball_Wall_Mount.stl"

    verts = read_binary_stl(stl_path)

    centerline = extract_centerline(verts, x_bins=200)
    tangents = compute_tangents(centerline)

    scale = 0.574  # same target scale as radial experiments

    r_orig, r_new, r_mean = compute_scaled_radii(
        verts,
        centerline,
        tangents,
        scale
    )

    print("\n[DIAGNOSTIC]")
    print(f"Original radius: min={r_orig.min():.2f}, max={r_orig.max():.2f}")
    print(f"Scaled radius:   min={r_new.min():.2f}, max={r_new.max():.2f}")

    # Spot-check a few slices along the sweep
    for s in [50, 100, 150]:
        print(f"\n[SLICE {s}]")
        print(f"  mean radius = {r_mean[s]:.2f}")


if __name__ == "__main__":
    main()