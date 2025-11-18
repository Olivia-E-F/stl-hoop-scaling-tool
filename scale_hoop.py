import struct        # For unpacking binary STL data (STL is a binary format)
import numpy as np   # For handling large arrays and easy math on vertices


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


def scale_hoop_region(data, x_cut=20.0, z_cut=130.0, scale=0.6):
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
        return 0, np.nan, np.nan

    # Compute geometric center of the hoop in y-z plane
    # These are the points we scale *around*
    cy = y[mask].mean()
    cz = z[mask].mean()

    # Scale only y and z positions of the masked vertices
    # Move them toward/away from the center by the scale factor
    y[mask] = cy + scale * (y[mask] - cy)
    z[mask] = cz + scale * (z[mask] - cz)

    # Put modified vertices back into the structured array shape
    data["verts"] = verts.reshape(data["verts"].shape)

    # Return how many points we changed + center used
    return mask.sum(), cy, cz


def main():
    """
    We Run!
    Reads STL → scales hoop → writes output file.
    """
    input_path = "Sports_Ball_Wall_Mount.stl"                # Original STL
    output_path = "Sports_Ball_Wall_Mount_scaled.stl"       # Edit name if you would like to test scaling

    # Read the STL binary content
    header, data = read_binary_stl(input_path)

    # Apply scaling to hoop region
    count, cy, cz = scale_hoop_region(
        data,
        x_cut=20.0,    # Where hoop starts in x-direction
        z_cut=130.0,   # Rim height threshold
        scale=0.6,     # 40% shrink (you can change this)
    )

    # Write out a new STL file with modified vertices
    write_binary_stl(output_path, header, data)

    # Printing notes with confirmation
    print(f"Scaled {count} vertices in hoop region.")
    print(f"Hoop center approx: y={cy:.3f}, z={cz:.3f}")
    print(f"Wrote: {output_path}")


# If you run:   python scale_hoop.py
# This block executes.
if __name__ == "__main__":
    main()
