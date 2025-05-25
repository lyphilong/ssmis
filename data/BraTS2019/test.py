import nibabel as nib
import h5py, numpy as np, pathlib, tqdm, os

from pathlib import Path
import h5py, random, torch



root  = Path("/home/admin2022/ssd1/AE_Long/zim-research/data/DyCON/data/BraTS2019/data")        # your .h5 folder
bad  = []

for p in tqdm.tqdm(sorted(root.glob("*.h5")), desc="checking"):
    with h5py.File(p, "r") as h:
        img, lbl = h["image"].shape, h["label"].shape
    if img[1:] != lbl:            # ignore channel dim
        bad.append((p.name, img, lbl))

print("\nBad files :", len(bad))
for f, i, l in bad[:20]:          # print first 20, if any
    print(f"{f:30}  image {i}  label {l}")


