
import h5py
import shutil
import os
import argparse

''' This script modifies the WFN.h5 file by copying the /mf_header from source_header_file to base_file.
It makes a copy of base_file and overwrites it with the new /mf_header.
This is useful for make similar but not equal WFN.h5 compatible with each other so BerkeleyGW doesn't give errors
when reading them such as: ERROR: eqpcor mean-field energy mismatch, etc

Usage:

python modify_WFN_header.py source_header_file base_file --output output_file
'''

parser = argparse.ArgumentParser(description="Replace /mf_header in a WFN.h5 file.")
parser.add_argument("source_header_file", help="WFN.h5 file to copy /mf_header from")
parser.add_argument("base_file", help="WFN.h5 file to apply the header to")
parser.add_argument("--output", default="WFN_mod.h5", dest="output_file", help="Output file name (default: WFN_fi_mod.h5)")
args = parser.parse_args()

source_header_file = args.source_header_file
base_file = args.base_file
output_file = args.output_file

# Step 1: Copy the entire original file to the new one
shutil.copy(base_file, output_file)
print(f"Copied {base_file} to {output_file}")

# Step 2: Replace /mf_header in the new file
with h5py.File(source_header_file, "r") as src, h5py.File(output_file, "a") as dst:
    # Remove old /mf_header if it exist
    if "/mf_header" in dst:
        del dst["/mf_header"]
        print("Deleted existing /mf_header in output file.")

    # Copy new /mf_header from source
    src.copy("/mf_header", dst)
    print("Copied new /mf_header from", source_header_file)
