#!/usr/bin/env python3
"""
Blob Detection for Nanoparticle Tracking
Uses scikit-image blob_log detector without deep learning models
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tifffile
from pathlib import Path
from typing import Tuple, List, Dict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.feature import blob_log
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class BlobDetectorUpscaled:
    """Simple blob detector with upscaling"""
    
    def __init__(
        self,
        upscale_factor: float = 4.0,
        blob_threshold: float = 0.05,
        min_sigma: float = 2,
        max_sigma: float = 20,
    ):
        self.upscale_factor = upscale_factor
        self.blob_threshold = blob_threshold
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        print(f"\n--- Initializing Blob Detector ---")
        print(f"  Upscaling: {upscale_factor}x")
        print(f"  Blob threshold: {blob_threshold}")
        print(f"  Sigma range: {min_sigma}-{max_sigma}")
    
    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        if self.upscale_factor == 1.0:
            return image
        h, w = image.shape[:2]
        new_h = int(h * self.upscale_factor)
        new_w = int(w * self.upscale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect blobs and create instance mask
        
        Returns:
            (instance_mask, num_particles) at original resolution
        """
        original_shape = image.shape[:2]
        
        # 1. Upscale
        upscaled = self.upscale_image(image)
        
        # 2. Normalize
        img_norm = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min())
        
        # 3. Detect blobs
        blobs = blob_log(
            img_norm,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=10,
            threshold=self.blob_threshold,
            overlap=0.5
        )
        
        if len(blobs) == 0:
            return np.zeros(original_shape, dtype=np.uint16), 0
        
        # 4. Create mask from blobs at upscaled resolution
        mask_upscaled = np.zeros(upscaled.shape[:2], dtype=np.uint16)
        
        for label_id, (y, x, r) in enumerate(blobs, start=1):
            # Create circular mask for each blob
            rr, cc = np.ogrid[:upscaled.shape[0], :upscaled.shape[1]]
            circle_mask = (rr - y)**2 + (cc - x)**2 <= (r * np.sqrt(2))**2
            mask_upscaled[circle_mask] = label_id
        
        # 5. Downscale mask to original resolution
        mask_original = cv2.resize(
            mask_upscaled.astype(np.float32),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint16)
        
        return mask_original, len(blobs)


# =====================
# Z-STACK UTILITIES
# =====================

def load_zstack(image_path: str) -> np.ndarray:
    print(f"Reading file: {image_path}...", end="", flush=True)
    img = tifffile.imread(image_path)
    print(" Done.")
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    print(f"  Shape: {img.shape}, Dtype: {img.dtype}")
    return img

def get_centroids_from_labels(labels: np.ndarray) -> List[Tuple[float, float]]:
    props = measure.regionprops(labels.astype(np.int32))
    return [(float(p.centroid[0]), float(p.centroid[1])) for p in props]

def get_particles_per_slice(masks_3d: np.ndarray):
    centroids_per_slice = []
    counts = []
    for z in range(masks_3d.shape[0]):
        cents = get_centroids_from_labels(masks_3d[z])
        centroids_per_slice.append(cents)
        counts.append(len(cents))
    return centroids_per_slice, counts

def link_particles_across_slices(centroids_per_slice, max_distance=10.0):
    tracks = {}
    next_tid = 1
    active = {}
    
    for z, cents in enumerate(centroids_per_slice):
        if not cents:
            active = {}
            continue
            
        curr_pos = np.array(cents)
        
        if not active:
            for y, x in cents:
                tracks[next_tid] = [(z, y, x)]
                active[next_tid] = (y, x)
                next_tid += 1
            continue

        prev_pos = np.array(list(active.values()))
        tids = list(active.keys())
        
        D = cdist(prev_pos, curr_pos)
        valid = D < max_distance
        
        matched_tracks = set()
        matched_dets = set()
        
        if valid.any():
            row_ind, col_ind = linear_sum_assignment(np.where(valid, D, 1e10))
            for r, c in zip(row_ind, col_ind):
                if D[r, c] < max_distance:
                    tid = tids[r]
                    y, x = curr_pos[c]
                    tracks[tid].append((z, float(y), float(x)))
                    active[tid] = (float(y), float(x))
                    matched_tracks.add(tid)
                    matched_dets.add(c)
        
        active = {t: pos for t, pos in active.items() if t in matched_tracks}
        
        for idx, (y, x) in enumerate(cents):
            if idx not in matched_dets:
                tracks[next_tid] = [(z, float(y), float(x))]
                active[next_tid] = (float(y), float(x))
                next_tid += 1
                
    return tracks

def filter_short_tracks(tracks, min_length=3):
    return {tid: tr for tid, tr in tracks.items() if len(tr) >= min_length}

def visualize_blob_zstack(img_zyx: np.ndarray, masks_3d: np.ndarray, centroids_per_slice, tracks, save_path: str, sample_interval: int = 5):
    """Visualize blob detection results matching MEDIAR style"""
    n_slices = img_zyx.shape[0]
    show_idx = np.arange(0, n_slices, sample_interval)
    if len(show_idx) > 10:
        show_idx = show_idx[::max(1, len(show_idx) // 10)]
    show_idx = show_idx[:10]
    
    fig, axes = plt.subplots(3, len(show_idx), figsize=(3*len(show_idx), 9))
    if len(show_idx) == 1:
        axes = axes[:, np.newaxis]
    
    for i, z in enumerate(show_idx):
        # Row 0: Original image
        axes[0, i].imshow(img_zyx[z], cmap='gray')
        axes[0, i].set_title(f'Z={z}')
        axes[0, i].axis('off')
        
        # Row 1: Blob detection overlay
        axes[1, i].imshow(img_zyx[z], cmap='gray', alpha=0.6)
        axes[1, i].imshow(masks_3d[z], cmap='tab20', alpha=0.4, vmin=0, vmax=20)
        axes[1, i].set_title(f'Detected: {len(centroids_per_slice[z])}')
        axes[1, i].axis('off')
        
        # Row 2: Tracking overlay
        axes[2, i].imshow(img_zyx[z], cmap='gray', alpha=0.5)
        axes[2, i].imshow(masks_3d[z], cmap='tab20', alpha=0.5, vmin=0, vmax=20)
        for tid, positions in tracks.items():
            for pz, py, px in positions:
                if pz == z:
                    axes[2, i].plot(px, py, 'r.', markersize=3)
        axes[2, i].set_title('Tracked')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Blob Detection Particle Tracking")
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to TIFF file or directory')
    parser.add_argument('--output', '-o', type=str, default='training_data_output/blob', help='Output directory')
    parser.add_argument('--upscale', '-u', type=float, default=4.0, help='Upscale factor')
    parser.add_argument('--blob-threshold', type=float, default=0.05, help='Blob detection threshold')
    parser.add_argument('--min-sigma', type=float, default=2, help='Minimum blob size (sigma)')
    parser.add_argument('--max-sigma', type=float, default=20, help='Maximum blob size (sigma)')
    parser.add_argument('--max-link-distance', type=float, default=10.0, help='Max distance for tracking')
    parser.add_argument('--min-track-length', type=int, default=3, help='Minimum track length')
    
    args = parser.parse_args()

    # File resolution
    if os.path.isfile(args.image):
        paths = [args.image]
    elif os.path.isdir(args.image):
        paths = [str(Path(args.image) / f) for f in os.listdir(args.image) 
                 if f.endswith(('.tif', '.tiff')) and not f.startswith('.')]
    else:
        print("Invalid input path.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Init detector
    detector = BlobDetectorUpscaled(
        upscale_factor=args.upscale,
        blob_threshold=args.blob_threshold,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma
    )

    summary = []
    
    # Process
    for p in paths:
        print(f"\nProcessing {Path(p).name}...")
        try:
            stack = load_zstack(p)
        except Exception as e:
            print(f"Could not load {p}: {e}")
            continue

        masks_3d = np.zeros_like(stack, dtype=np.uint16)
        
        print(f"Detecting blobs on {stack.shape[0]} slices...")
        for z in tqdm(range(stack.shape[0]), desc="Processing"):
            masks_3d[z], count = detector.predict(stack[z])

        print(f"\nLinking particles across slices...")
        centroids_per_slice, counts = get_particles_per_slice(masks_3d)
        tracks = link_particles_across_slices(centroids_per_slice, max_distance=args.max_link_distance)
        tracks = filter_short_tracks(tracks, min_length=args.min_track_length)
        print(f"Found {len(tracks)} tracks.")
        
        # Save
        base_name = Path(p).stem
        out_sub = Path(args.output) / base_name
        out_sub.mkdir(parents=True, exist_ok=True)
        
        # Save masks
        mask_path = out_sub / f"{base_name}_blob_masks.tif"
        tifffile.imwrite(str(mask_path), masks_3d.astype(np.uint16))
        
        # Save visualization
        viz_path = out_sub / f"{base_name}_blob_viz.png"
        visualize_blob_zstack(stack, masks_3d, centroids_per_slice, tracks, 
                             save_path=str(viz_path), sample_interval=5)
        
        summary.append((Path(p).name, len(tracks)))
        print(f"Results saved to {out_sub}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, count in summary:
        print(f"{name}: {count} tracks")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
