import struct        # For unpacking binary STL data (STL is a binary format)
import numpy as np   # For handling large arrays and easy math on vertices
import os            # NEW: required for versioned filenames


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
            ("normal", np.float32, (3,)),  # normal vector (3 floats)
            ("verts",  np.float32, (3, 3)), # 3 vertices, each with x,y,z (9 floats)
            ("attr",   np.uint16)          # attribute byte count (usually zero)
        ])

        # Load all triangles into a numpy structured array
        data = np.fromfile(f, dtype=tri_dtype, count=tri_count)

    return header, data


def write_binary_stl(path, header, data):
    """
    Write a binary STL file with the given header + triangles.
    """
    with open(path, "wb") as f:   # Write binary file
        # Ensure header is exactly 80 bytes (pad if too short)
        if len(header) < 80:
            header = header + b" " * (80 - len(header)) # b" "" = byte space
        f.write(header[:80])      # Write header (80 bytes)

        # Write number of triangles
        f.write(struct.pack("<I", len(data)))

        # Write triangle blocks directly from numpy structured array
        data.tofile(f)


# Scale factor = New/old, for me 3
def scale_hoop_region(data, x_cut=20.0, z_cut=130.0, scale=0.288):
    """
    Modify only the hoop region of the mesh (not the whole STL).
      - x < x_cut  → points are on the hoop side, not the wall mount
      - z >= z_cut → points are near the top/outer rim
    """

    # Flatten vertices from shape (N_triangles, 3 vertices, 3 coords)
    # into shape (N_vertices, 3)
    verts = data["verts"].reshape(-1, 3)

    # Extract x, y, z columns for easier reading
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # Mask selecting only vertices in the hoop region
    mask = (x < x_cut) & (z >= z_cut)

    # Mask cannot be empty for scaling
    if not np.any(mask):
        print("Warning: hoop mask caught no vertices.")
        return 0, np.nan, np.nan, np.nan, np.nan

    # Copy original hoop vertices BEFORE scaling
    orig_y = y[mask].copy()
    orig_z = z[mask].copy()

    # Compute geometric center of the hoop in y-z plane
    # These are the points we scale *around*
    # Compute original center and “radius”
    cy = orig_y.mean()
    cz = orig_z.mean()
    orig_dist = np.sqrt((orig_y - cy)**2 + (orig_z - cz)**2)
    orig_radius = orig_dist.mean()

    # Scale only y and z positions of the masked vertices
    # Move them toward/away from the center by the scale factor
    y[mask] = cy + scale * (y[mask] - cy)
    z[mask] = cz + scale * (z[mask] - cz)

    # Compute radius AFTER scaling to compare
    new_dist = np.sqrt((y[mask] - cy)**2 + (z[mask] - cz)**2)
    new_radius = new_dist.mean()

    # Put modified vertices back into the structured array shape
    data["verts"] = verts.reshape(data["verts"].shape)

    # Return how many points we changed + center used
    return mask.sum(), cy, cz, orig_radius, new_radius


# NEW: compute inches-per-CAD-unit from a test print
def compute_inches_per_cad(print_diameter_in, cad_radius):
    return (print_diameter_in / 2) / cad_radius


# NEW: compute final STL scale factor based on desired real size
def compute_final_scale(original_real_diameter_in, target_diameter_in):
    return target_diameter_in / original_real_diameter_in


# NEW: auto-name output STL by diameter + scale
def build_output_name(target_diameter_in, scale):
    return f"hoop_{target_diameter_in:.2f}in_scale{scale:.3f}.stl"


# NEW: automatic versioning (never overwrite)
def get_versioned_filename(base_name):
    name, ext = os.path.splitext(base_name)
    version = 1
    candidate = base_name

    while os.path.exists(candidate):
        version += 1
        candidate = f"{name}_v{version}{ext}"

    return candidate


def main():
    """
    We Run!
    Reads STL → scales hoop → writes output file.
    """
    input_path = "Sports_Ball_Wall_Mount.stl"  # Original STL

    # NEW: Your measured test print values
    test_print_diameter_in = 6.25     # physical print size of scaled model
    test_cad_radius = 5.134           # CAD radius of that version
    orig_cad_radius = 8.556           # CAD radius of original STL

    # convert CAD units → inches using physical print
    inches_per_cad = compute_inches_per_cad(test_print_diameter_in, test_cad_radius)

    # compute original STL’s real-world size
    original_real_diameter_in = 2 * orig_cad_radius * inches_per_cad

    # target hoop real-world size
    target_diameter_in = 3.0

    # compute correct scale factor
    final_scale = compute_final_scale(original_real_diameter_in, target_diameter_in)

    # Read the STL binary content
    header, data = read_binary_stl(input_path)

    # Apply scaling to hoop region
    count, cy, cz, orig_r, new_r = scale_hoop_region(
        data,
        x_cut=20.0,
        z_cut=130.0,
        scale=final_scale,   # real-world-correct scale
    )

    # dynamic + versioned output filename
    output_filename = build_output_name(target_diameter_in, final_scale)
    output_path = get_versioned_filename(output_filename)

    # Write out a new STL file with modified vertices
    write_binary_stl(output_path, header, data)

    # Printing notes with confirmation
    print(f"Original hoop radius: {orig_r:.3f}")
    print(f"New hoop radius:      {new_r:.3f}")
    print(f"Observed scale factor: {new_r / orig_r:.3f}")
    print(f"1 CAD unit = {inches_per_cad:.4f} inches")
    print(f"Original STL diameter ≈ {original_real_diameter_in:.3f} inches")
    print(f"Final applied scale = {final_scale:.4f}")
    print(f"Scaled {count} vertices in hoop region.")
    print(f"Hoop center approx: y={cy:.3f}, z={cz:.3f}")
    print(f"Wrote: {output_path}")


# If you run:   python scale_hoop.py
# This block executes.
if __name__ == "__main__":
    main()