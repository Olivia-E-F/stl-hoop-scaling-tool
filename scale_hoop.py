"""
Selective STL hoop scaling.

This script loads a binary STL model, identifies the hoop portion of the mesh
using coordinate-based filtering, scales that region in the Y–Z plane, and
writes out a modified STL file. The mounting structure is left unchanged.
"""

import struct
import numpy as np


# ---------------------------------------------------------------------------
# read_binary_stl(path)
# ---------------------------------------------------------------------------
# Reads a **binary STL** file and returns:
#   - header:   the 80-byte STL header
#   - normals:  (N, 3) float32 array of triangle normals
#   - triangles: (N, 3, 3) float32 array of triangle vertex coordinates
#
# Binary STL format summary:
#   Bytes 0–79   → header (80 bytes)
#   Bytes 80–83  → uint32 triangle count
#   For each triangle (50 bytes total):
#       3 floats → normal vector (nx, ny, nz)
#       3 floats → vertex 1  (x1, y1, z1)
#       3 floats → vertex 2  (x2, y2, z2)
#       3 floats → vertex 3  (x3, y3, z3)
#       2 bytes  → attribute (ignored)
#
# We use struct.unpack("<12fH", ...) because:
#    "<"  = little-endian (STL spec)
#    "12f" = 12 float32 values (normal + 3 vertices)
#    "H"  = uint16 (attribute)
# ---------------------------------------------------------------------------

def read_binary_stl(path):
    with open(path, "rb") as f:
        header = f.read(80)                      # STL header text
        tri_count = struct.unpack("<I", f.read(4))[0]   # number of triangles
        records = f.read()                       # remaining binary payload

    # Each triangle occupies 50 bytes in binary STL
    record_size = 50

    # Pre-allocate arrays (faster and cleaner than appending)
    normals = np.empty((tri_count, 3), dtype=np.float32)
    triangles = np.empty((tri_count, 3, 3), dtype=np.float32)

    offset = 0
    for i in range(tri_count):
        block = records[offset : offset + record_size]

        # Unpack 12 floats + attribute
        floats = struct.unpack("<12fH", block)

        # First 3 floats → triangle normal
        normals[i] = floats[0:3]

        # Next 9 floats → vertices (3 floats per vertex)
        triangles[i, 0, :] = floats[3:6]
        triangles[i, 1, :] = floats[6:9]
        triangles[i, 2, :] = floats[9:12]

        offset += record_size

    return header, normals, triangles



# ---------------------------------------------------------------------------
# write_binary_stl(path, header, normals, triangles)
# ---------------------------------------------------------------------------
# Writes the STL back out in proper **binary STL** format.
# We preserve the original normals exactly (recomputing normals is optional
# but unnecessary here).
#
# The structure written mirrors the read format.
# ---------------------------------------------------------------------------
def write_binary_stl(path, header, normals, triangles):
    with open(path, "wb") as f:
        # Ensure header is exactly 80 bytes (pad if necessary)
        f.write(header[:80].ljust(80, b' '))

        # Write number of triangles
        f.write(struct.pack("<I", triangles.shape[0]))

        # Write each triangle record
        for i in range(triangles.shape[0]):
            n = normals[i]
            v1, v2, v3 = triangles[i]

            record = struct.pack(
                "<12fH",
                # Normal vector
                float(n[0]), float(n[1]), float(n[2]),
                # Vertices
                float(v1[0]), float(v1[1]), float(v1[2]),
                float(v2[0]), float(v2[1]), float(v2[2]),
                float(v3[0]), float(v3[1]), float(v3[2]),
                # Attribute byte (unused)
                0
            )
            f.write(record)



# ---------------------------------------------------------------------------
# scale_hoop_in_file()
# ---------------------------------------------------------------------------
# Performs the region-based scaling.
#
# Steps:
#   1. Load STL and flatten vertices into (N*3, 3)
#   2. Create a boolean mask selecting the hoop region
#   3. Compute geometric center of hoop in Y–Z plane
#   4. Scale only the Y and Z coordinates toward (cy, cz)
#   5. Reassemble triangles and write new STL
#
# We only sclae Y and Z; scaling X would distort the depth.
#
# scale_factor:
#   0.6 = shrink by 40%
#   1.0 = no change
#   1.2 = enlarge by 20%
# ---------------------------------------------------------------------------
def scale_hoop_in_file(input_stl, output_stl, scale_factor=0.6):
    # 1. Load full STL geometry
    header, normals, triangles = read_binary_stl(input_stl)

    # Convert triangles → flat array of all vertices (easier to mask)
    verts = triangles.reshape(-1, 3)

    # Extract coordinate columns for clarity
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # ---------------------------------------------------------------
    # 2. Identify the hoop region
    #
    # These thresholds come from an analysis of the specific STL:
    #   - Hoop vertices occur near the *front* (x < 20)
    #   - Hoop vertices occur in the *upper ring* (z >= 130)
    #
    # This mask selects exactly those vertices.
    # ---------------------------------------------------------------
    hoop_mask = (x < 20.0) & (z >= 130.0)

    if hoop_mask.sum() == 0:
        raise RuntimeError(
            "No hoop vertices found. Thresholds may need adjustment."
        )

    # ---------------------------------------------------------------
    # 3. Compute hoop center (cy, cz) in Y–Z plane
    #
    # We scale relative to this center so the hoop shrinks or expands
    # uniformly rather than drifting off-axis.
    # ---------------------------------------------------------------
    cy = y[hoop_mask].mean()
    cz = z[hoop_mask].mean()

    # ---------------------------------------------------------------
    # 4. Apply scale transform ONLY to hoop vertices
    #
    # Formula:
    #   new_y = cy + scale_factor * (old_y - cy)
    #   new_z = cz + scale_factor * (old_z - cz)
    #
    # This pulls points toward (cy, cz) when scale_factor < 1.
    # X is unchanged to preserve wall-mount depth.
    # ---------------------------------------------------------------
    y[hoop_mask] = cy + scale_factor * (y[hoop_mask] - cy)
    z[hoop_mask] = cz + scale_factor * (z[hoop_mask] - cz)

    # Reassemble vertices back into triangle array shape
    triangles_scaled = verts.reshape(triangles.shape)

    # 5. Write new STL
    write_binary_stl(output_stl, header, normals, triangles_scaled)

    # Return information for debugging or logging
    return cy, cz, hoop_mask.sum()



# ---------------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------------
# When this file is run directly (`python3 scale_hoop.py`), we:
#   • Load the input STL in the same folder
#   • Apply 40% shrink (scale_factor = 0.6)
#   • Output a new scaled STL next to it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_path  = "Sports_Ball_Wall_Mount.stl"
    output_path = "Sports_Ball_Wall_Mount_hoop_scaled_40pct_smaller.stl"

    cy, cz, count = scale_hoop_in_file(
        input_path,
        output_path,
        scale_factor=0.6
    )

    print("DONE!")
    print(f"Hoop center: y = {cy:.4f}, z = {cz:.4f}")
    print(f"Vertices scaled: {count}")
    print(f"Output file: {output_path}")
