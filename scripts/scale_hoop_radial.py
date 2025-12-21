# Depenndencies and Setup
import struct        # For unpacking binary STL data (STL is a binary format)
import numpy as np   # For handling large arrays and easy math on vertices
import os            # NEW: required for versioned filenames

# =================================
# STL IO/Utilities
# =================================

def read_binary_stl(path):
    """
    Read a binary STL file and return:
    - header (80 bytes)
    - structured array of triangles (normals, verts, attr)
    """
    with open(path, "rb") as f:       # Open the file in binary mode
        header = f.read(80)           # First 80 bytes = STL header (not used, but must be kept)
        tri_count = struct.unpack("<I", f.read(4))[0]  # Next 4 bytes = number of triangles

        # Each triangle in binary STL is:
        #   - 3 floats for normal (12 bytes)
        #   - 3 vertices × 3 floats each (36 bytes)
        #   - 2-byte attribute field
        # Total = 50 bytes per triangle
        tri_dtype = np.dtype([
            ("normal", np.float32, (3,)),   # normal vector (3 floats)
            ("verts",  np.float32, (3, 3)), # 3 vertices, each with x,y,z (9 floats)
            ("attr",   np.uint16)            # attribute byte count (usually zero)
        ])

        # Load all triangles into a numpy structured array
        data = np.fromfile(f, dtype=tri_dtype, count=tri_count)

    return header, data

# STL Writer
def write_binary_stl(path, header, data):
    """
    Write a binary STL file with the given header + triangles.
    """
    with open(path, "wb") as f:   # Write binary file
        # Ensure header is exactly 80 bytes (pad if too short)
        if len(header) < 80:
            header = header + b" " * (80 - len(header))
        f.write(header[:80])      # Write header (80 bytes)

        # Write number of triangles
        f.write(struct.pack("<I", len(data)))

        # Write triangle blocks directly from numpy structured array
        data.tofile(f)

# =================================
# Geometry Transform
# =================================

## ** Scale factor still an issue
def scale_hoop_region(data, x_cut=20.0, z_cut=130.0, scale=1.0):
    """
    Modify only the hoop region of the mesh (not the whole STL).
      - x < x_cut  → points are on the hoop side, not the wall mount
      - z >= z_cut → points are near the top/outer rim
    """

    # Flatten vertices from shape (N_triangles, 3 vertices, 3 coords)
    verts = data["verts"].reshape(-1, 3)

    # Extract x, y, z columns for easier reading
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # ** CHANGED: snapshot vertices for proof of movement
    verts_before = verts.copy()  # **

    # ** CHANGED: estimate true hoop center via least-squares circle fit
    A = np.column_stack([2*y, 2*z, np.ones_like(y)])  # **
    b = y*y + z*z                                    # **
    cy, cz, _ = np.linalg.lstsq(A, b, rcond=None)[0] # **


    # ** CHANGED: radial distance in hoop plane (Y–Z)
    r = np.sqrt((y - cy)**2 + (z - cz)**2)  # **

    # ** CHANGED: select outer radial band instead of axis-aligned box
    r_thresh = np.percentile(r, 85)  # ** outer rim band
    mask = r >= r_thresh  # **

    # ** CHANGED: mask must not be empty — fail hard
    if not np.any(mask):  # **
        raise RuntimeError("Hoop mask caught ZERO vertices — aborting.")  # **

    # Copy original hoop vertices BEFORE scaling
    orig_y = y[mask].copy()
    orig_z = z[mask].copy()

    orig_dist = np.sqrt((orig_y - cy)**2 + (orig_z - cz)**2)
    orig_radius = orig_dist.mean()

    # Scale only y and z positions of the masked vertices
    y[mask] = cy + scale * (y[mask] - cy)
    z[mask] = cz + scale * (z[mask] - cz)

    # Compute radius AFTER scaling to compare
    new_dist = np.sqrt((y[mask] - cy)**2 + (z[mask] - cz)**2)
    new_radius = new_dist.mean()

    # Put modified vertices back into the structured array shape
    data["verts"] = verts.reshape(data["verts"].shape)

    # ** CHANGED: PROOF — count how many vertices actually moved
    delta = np.linalg.norm(verts - verts_before, axis=1)  # **
    moved = np.count_nonzero(delta > 1e-6)  # **
    print(f"[DEBUG] Vertices moved: {moved}")  # **

    # ** CHANGED: refuse silent failure
    assert moved > 0, "No vertices moved — geometry unchanged"  # **

    # Return how many points we changed + center used
    return mask.sum(), cy, cz, orig_radius, new_radius

# =================================
# Output Naming
# =================================

# Auto-name output STL by diameter + scale
def build_output_name(target_diameter_in, scale):
    return f"hoop_{target_diameter_in:.2f}in_scale{scale:.3f}.stl"


# Auto-Versioning
def get_versioned_filename(base_name):
    name, ext = os.path.splitext(base_name)
    version = 1
    candidate = base_name

    while os.path.exists(candidate):
        version += 1
        candidate = f"{name}_v{version}{ext}"

    return candidate

# =================================
# Main Pipeline 
# =================================

def main():
    """
    We Run!
    Reads STL → scales hoop → writes output file.
    """

    # =================================
    # Input Config
    # =================================

    # CHANGED:
    # Use the physically printed 6-inch STL as the canonical source
    input_path = "baseline_hoop.stl"

    # =================================
    # Read STL
    # =================================

    header, data = read_binary_stl(input_path)

    # =================================
    # Measurement (relative, unitless)
    # =================================

    _, _, _, orig_r, _ = scale_hoop_region(
        data,
        x_cut=20.0,
        z_cut=130.0,
        scale=1.0
    )

    orig_diameter_units = 2 * orig_r

    print("\n[MEASUREMENT]")
    print(f"Hoop Diameter: {orig_diameter_units:.3f} STL Units")

    # =================================
    # Scaling Decision (THIS IS THE KNOB)
    # =================================

    # Known physical size of this STL (measured print)
    baseline_diameter_in = 6.10

    # Desired target size
    target_diameter_in = 3.50

    # Relative scale ONLY — no unit conversion
    final_scale = target_diameter_in / baseline_diameter_in

    print("\n[SCALING]")
    print(f"Baseline diameter: {baseline_diameter_in:.2f} in")
    print(f"Target diameter:   {target_diameter_in:.2f} in")
    print(f"Scale factor:      {final_scale:.4f}")

    # =================================
    # Apply Scaling
    # =================================

    count, cy, cz, orig_r, new_r = scale_hoop_region(
        data,
        x_cut=20.0,
        z_cut=130.0,
        scale=final_scale
    )

    # =================================
    # Write Output
    # =================================

    output_filename = build_output_name(target_diameter_in, final_scale)
    output_path = get_versioned_filename(output_filename)

    write_binary_stl(output_path, header, data)

    print("\n[RESULT]")
    print(f"Scaled {count} vertices in hoop region.")
    print(f"Hoop center approx: y={cy:.3f}, z={cz:.3f}")
    print(f"Wrote: {output_path}")


# If you run:   python scale_hoop.py
# This block executes.
if __name__ == "__main__":
    main()
