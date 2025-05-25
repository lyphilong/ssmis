import nibabel as nib
import h5py, numpy as np, pathlib, tqdm, os

# ─── paths ──────────────────────────────────────────────────────────────
# Adding both HGG and LGG paths
HGG_ROOT = pathlib.Path("./MICCAI_BraTS_2019_Data_Training/HGG")
LGG_ROOT = pathlib.Path("./MICCAI_BraTS_2019_Data_Training/LGG")
DST_ROOT = pathlib.Path("./data")    # <<< EDIT ME
# ────────────────────────────────────────────────────────────────────────
DST_ROOT.mkdir(parents=True, exist_ok=True)

# Modalities in BraTS dataset
# "0": "FLAIR" - T2-weighted-Fluid-Attenuated Inversion Recovery MRI (used for whole tumor segmentation)
# "1": "T1w" - T1-weighted MRI
# "2": "t1ce" - T1-gadolinium contrast MRI
# "3": "T2w" - T2-weighted MRI
MODS = ['flair', 't1', 't1ce', 't2']
FLAIR_IDX = 0  # Index of FLAIR in the MODS list

# Labels in BraTS dataset
# "0": "background" - No tumor
# "1": "edema" - Swelling around tumor
# "2": "non-enhancing tumor" - Tumor that isn't enhanced by Gadolinium contrast
# "3": "enhancing tumour" - Gadolinium contrast enhanced regions

def preprocess_labels(seg):
    """
    Process the ground truth labels for BraTS dataset
    For 'whole tumor' segmentation, combine all non-zero labels into one class
    
    Args:
        seg: The segmentation mask
        
    Returns:
        Binary segmentation mask (0=background, 1=tumor)
    """
    # For whole tumor binary segmentation, combine all tumor regions (classes 1,2,3) into one class (1)
    binary_seg = seg.copy()
    binary_seg[binary_seg > 0] = 1  # Combine all tumor regions into a single class
    
    return binary_seg

def normalize_img(img):
    """
    Normalize the pixel values with Z-score normalization -1, 1
    
    Args:
        img: Input image
        
    Returns:
        Normalized image
    """
    return (img - img.mean()) / img.std()

def process_cases(root_path, case_prefix="HGG_"):
    """
    Process cases from a specific path
    
    Args:
        root_path: Path to the cases
        case_prefix: Prefix to add to case names to distinguish HGG vs LGG
        
    Returns:
        Number of cases processed
    """
    cases = sorted([p for p in root_path.iterdir() if p.name.startswith("BraTS19")])
    
    for case_dir in tqdm.tqdm(cases, desc=f"Converting from {root_path.name}"):
        # read the FLAIR modality (for whole tumor segmentation)
        flair_img = nib.load(case_dir / f"{case_dir.name}_flair.nii").get_fdata(dtype=np.float32)
        
        # read the segmentation mask
        seg = nib.load(case_dir / f"{case_dir.name}_seg.nii").get_fdata().astype(np.uint8)
        
        # Process the segmentation mask to create binary tumor mask (whole tumor)
        binary_seg = preprocess_labels(seg)
        
        # Normalize the FLAIR image
        flair_norm = normalize_img(flair_img)
        # print(flair_norm.shape)
        
        # # Add an axis for the channel dimension
        # flair_norm = np.expand_dims(flair_norm, axis=0)
        
        # write one HDF5 per patient
        out = DST_ROOT / f"{case_dir.name}.h5"
        with h5py.File(out, "w") as h5:
            h5.create_dataset("image", data=flair_norm, compression="gzip", chunks=True)
            h5.create_dataset("label", data=binary_seg, compression="gzip", chunks=True)
            
            # Store metadata
            h5.attrs['tumor_type'] = root_path.name  # HGG or LGG
            h5.attrs['case_name'] = case_dir.name
            h5.attrs['modality'] = 'flair'
    
    return len(cases)

# Main processing
print(f"Starting BraTS dataset conversion for 'whole tumor' binary segmentation...")

# Process HGG cases
if HGG_ROOT.exists():
    hgg_count = process_cases(HGG_ROOT, case_prefix="HGG_")
    print(f"Processed {hgg_count} HGG cases")
else:
    print(f"Warning: HGG directory not found at {HGG_ROOT}")
    hgg_count = 0

# Process LGG cases
if LGG_ROOT.exists():
    lgg_count = process_cases(LGG_ROOT, case_prefix="LGG_")
    print(f"Processed {lgg_count} LGG cases")
else:
    print(f"Warning: LGG directory not found at {LGG_ROOT}")
    lgg_count = 0

print(f"Conversion complete. Total {hgg_count + lgg_count} cases processed.")
print(f"Files saved to {DST_ROOT}")
