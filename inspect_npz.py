import sys
import zipfile

import numpy as np


def _read_npy_header(npz_file, member_name):
    with npz_file.open(member_name) as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"Unsupported .npy version {version} for {member_name}")

    return shape, fortran_order, dtype


def inspect_npz(file_path):
    """
    Loads and prints the schema of the provided NPZ file without dereferencing
    object payloads that may import heavyweight native libraries.
    """
    print(f"Inspecting File: {file_path}")
    print("-" * 50)
    try:
        with zipfile.ZipFile(file_path) as zf:
            members = zf.namelist()
            print(f"Total keys found: {len(members)}")
            print("-" * 50)

            for member_name in members:
                key = member_name[:-4] if member_name.endswith(".npy") else member_name
                shape, _, dtype = _read_npy_header(zf, member_name)
                dtype_label = "object" if dtype.hasobject else str(dtype)
                suffix = " | Deferred object payload" if dtype.hasobject else ""
                print(f"Key: {key:20s} | Shape: {shape} | Dtype: {dtype_label}{suffix}")

            if "xyz.npy" in members:
                print("-" * 50)
                print("Statistics for 'xyz':")
                with zf.open("xyz.npy") as xyz_fh:
                    xyz = np.load(xyz_fh).astype(np.float32)
                print(f"  Min: {np.min(xyz, axis=0)}")
                print(f"  Max: {np.max(xyz, axis=0)}")

            print("-" * 50)
            print("Archive members:")
            for name in members:
                print(f"  {name}")

    except Exception as e:
        print(f"Failed to load or inspect {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    file_path = "point_cloud_pp.npz"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    inspect_npz(file_path)
