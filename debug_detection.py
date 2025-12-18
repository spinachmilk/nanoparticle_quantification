#!/usr/bin/env python3
"""
Diagnostic script to debug why Cellpose and SAM2 are not detecting particles.
Tests a single slice with all three methods and outputs detailed diagnostics.
"""

import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_image_properties(img):
    """Analyze basic image properties"""
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Min/Max: {img.min()}/{img.max()}")
    print(f"  Mean: {img.mean():.2f}, Std: {img.std():.2f}")
    print(f"  Median: {np.median(img):.2f}")
    
def test_cellpose_detection(img_slice, upscale_factor=4.0):
    """Test Cellpose with different parameters"""
    print("\n" + "="*60)
    print("CELLPOSE DETECTION TEST")
    print("="*60)
    
    try:
        from cellpose import models
    except ImportError:
        print("Cellpose not installed")
        return None
    
    # Original image
    print("\n1. Original Image:")
    analyze_image_properties(img_slice)
    
    # Upscale
    h, w = img_slice.shape[:2]
    new_h, new_w = int(h * upscale_factor), int(w * upscale_factor)
    upscaled = cv2.resize(img_slice, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    print(f"\n2. Upscaled Image ({upscale_factor}x):")
    analyze_image_properties(upscaled)
    
    # Test different diameters
    print("\n3. Testing Cellpose with different diameters...")
    model = models.CellposeModel(gpu=False, model_type='cyto2')
    
    test_diameters = [5, 15, 30, 60]  # Original pixels * upscale
    results = {}
    
    for diameter in test_diameters:
        scaled_diam = diameter * upscale_factor
        print(f"\n  Testing diameter={diameter} (scaled={scaled_diam:.1f})...")
        
        masks, flows, styles = model.eval(
            upscaled,
            diameter=scaled_diam,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        
        n_detected = len(np.unique(masks)) - 1  # Subtract background
        print(f"    Detected: {n_detected} particles")
        results[diameter] = (masks, n_detected)
    
    return results

def test_sam2_blob_detection(img_slice):
    """Test SAM2's blob detection (without full SAM2 model)"""
    print("\n" + "="*60)
    print("SAM2 BLOB DETECTION TEST")
    print("="*60)
    
    try:
        from skimage.feature import blob_log
    except ImportError:
        print("skimage not installed")
        return None
    
    print("\n1. Original Image:")
    analyze_image_properties(img_slice)
    
    # Normalize
    img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    
    print("\n2. Testing blob detection with different thresholds...")
    test_thresholds = [0.01, 0.02, 0.05, 0.1]
    results = {}
    
    for thresh in test_thresholds:
        print(f"\n  Threshold={thresh}:")
        blobs = blob_log(
            img_norm,
            min_sigma=2,
            max_sigma=20,
            num_sigma=10,
            threshold=thresh,
            overlap=0.5
        )
        print(f"    Detected: {len(blobs)} blobs")
        results[thresh] = blobs
    
    return results

def visualize_comparisons(img_slice, cellpose_results, sam2_results, output_path):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Cellpose results
    if cellpose_results:
        diams = list(cellpose_results.keys())[:2]
        for i, diam in enumerate(diams):
            if i < 2:
                masks, n = cellpose_results[diam]
                # Downscale mask back to original size
                masks_down = cv2.resize(masks.astype(np.float32), 
                                       (img_slice.shape[1], img_slice.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                axes[0, i+1].imshow(img_slice, cmap='gray', alpha=0.6)
                axes[0, i+1].imshow(masks_down, cmap='tab20', alpha=0.4)
                axes[0, i+1].set_title(f'Cellpose diam={diam}\n({n} detected)')
                axes[0, i+1].axis('off')
    
    # SAM2 blob detection results
    if sam2_results:
        thresholds = list(sam2_results.keys())[:3]
        for i, thresh in enumerate(thresholds):
            if i < 3:
                blobs = sam2_results[thresh]
                axes[1, i].imshow(img_slice, cmap='gray')
                if len(blobs) > 0:
                    # Plot detected blobs
                    for blob in blobs:
                        y, x, r = blob
                        c = plt.Circle((x, y), r*np.sqrt(2), color='red', 
                                     linewidth=2, fill=False)
                        axes[1, i].add_patch(c)
                axes[1, i].set_title(f'SAM2 Blobs thresh={thresh}\n({len(blobs)} detected)')
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

def main():
    # Load a test image
    data_dir = Path("/Users/cindyli/nanoparticle-tracking/training_data_cropped")
    tiff_files = list(data_dir.glob("*.tif"))
    
    if not tiff_files:
        print("No TIFF files found!")
        return
    
    test_file = tiff_files[0]
    print(f"Testing with: {test_file.name}")
    
    # Load z-stack and take middle slice
    stack = tifffile.imread(test_file)
    if stack.ndim == 2:
        img_slice = stack
    else:
        mid_z = stack.shape[0] // 2
        img_slice = stack[mid_z]
    
    print(f"\nTest slice shape: {img_slice.shape}")
    
    # Run tests
    cellpose_results = test_cellpose_detection(img_slice, upscale_factor=4.0)
    sam2_results = test_sam2_blob_detection(img_slice)
    
    # Visualize
    output_dir = Path("/Users/cindyli/nanoparticle-tracking/debug_output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"detection_comparison_{test_file.stem}.png"
    visualize_comparisons(img_slice, cellpose_results, sam2_results, output_path)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nRecommendations:")
    print("1. Check which diameter works best for Cellpose")
    print("2. Check which threshold works best for SAM2 blob detection")
    print("3. Verify image preprocessing matches working MEDIAR approach")

if __name__ == "__main__":
    main()
