"""
SAM 2-based Nanoparticle Segmentation with Image Upscaling
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
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

# Check for SAM 2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("ERROR: SAM 2 not installed.")
    print("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)


class SAM2PredictorUpscaled:
    """
    Wraps SAM 2 with automatic upscaling and blob-based prompt generation.
    """
    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: str,
        device: str = "mps",  # Changed default string to 'mps' to be explicit
        upscale_factor: float = 4.0,
        blob_threshold: float = 0.05,  # Optimized for TIFF uint16 images
    ):
        # 1. Device Selection
        self.device = device
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("  ! MPS requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Auto-fallback if 'auto' was passed
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.upscale_factor = upscale_factor
        self.blob_threshold = blob_threshold
        
        print(f"\n--- Initializing SAM 2 ---")
        print(f"  Config: {model_cfg}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Upscaling: {upscale_factor}x")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # 2. Load Model (Verbose)
        try:
            print("  Loading model architecture...", end="", flush=True)
            self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            print(" Done.")
            
            # MPS Stability Fix: Force float32
            if self.device == "mps":
                print("  Applying MPS stability fix (converting to float32)...", end="", flush=True)
                self.model.to(dtype=torch.float32)
                print(" Done.")
            
            print("  Initializing Image Predictor...", end="", flush=True)
            self.predictor = SAM2ImagePredictor(self.model)
            print(" Done.")
            
        except Exception as e:
            print(f"\nCRITICAL ERROR loading SAM 2: {e}")
            print("Ensure the .yaml config file is in the same folder or in the sam2/configs directory.")
            raise e

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        if self.upscale_factor == 1.0:
            return image
        h, w = image.shape[:2]
        new_h = int(h * self.upscale_factor)
        new_w = int(w * self.upscale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def downscale_mask(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        if self.upscale_factor == 1.0:
            return mask
        return cv2.resize(
            mask.astype(np.float32), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(mask.dtype)

    def generate_prompts(self, image_rgb: np.ndarray) -> np.ndarray:
        # Convert to grayscale for blob detection
        if len(image_rgb.shape) == 3:
            img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = image_rgb
            
        img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

        # Blob detection
        blobs = blob_log(
            img_norm,
            min_sigma=2,     
            max_sigma=20,    
            num_sigma=10,
            threshold=self.blob_threshold,
            overlap=0.5
        )

        if len(blobs) == 0:
            return np.array([])
            
        # Swap columns: [y, x] -> [x, y]
        return blobs[:, [1, 0]]

    def predict(self, image: np.ndarray, max_prompts: int = 100) -> Tuple[np.ndarray, int]:
        original_shape = image.shape[:2]

        # 1. Upscale
        upscaled = self.upscale_image(image)
        
        # 2. RGB Format
        if len(upscaled.shape) == 2:
            upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        elif upscaled.shape[2] == 1:
            upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        else:
            upscaled_rgb = upscaled
        
        # 3. uint8 conversion
        if upscaled_rgb.dtype != np.uint8:
            mi, ma = upscaled_rgb.min(), upscaled_rgb.max()
            if ma > mi:
                upscaled_rgb = (255 * (upscaled_rgb - mi) / (ma - mi)).astype(np.uint8)
            else:
                upscaled_rgb = upscaled_rgb.astype(np.uint8)

        # 4. Prompt Gen
        points = self.generate_prompts(upscaled_rgb)
        if len(points) == 0:
            return np.zeros(original_shape, dtype=np.uint16), 0

        # 5. Inference
        self.predictor.set_image(upscaled_rgb)
        
        composite_mask = np.zeros(upscaled_rgb.shape[:2], dtype=np.uint16)
        label_id = 1

        # Limit prompts for efficiency
        points_to_process = points[:max_prompts]
        
        for i, pt in enumerate(points_to_process):
            try:
                # Use multimask_output=True to get 3 mask options, pick the smallest good one
                masks, scores, _ = self.predictor.predict(
                    point_coords=pt[None, :],
                    point_labels=np.array([1]),
                    multimask_output=True  # Get multiple mask options
                )
                
                # Pick the best mask: highest score among reasonably-sized masks
                # Reject masks that are too large (likely whole-image masks)
                max_size = upscaled_rgb.shape[0] * upscaled_rgb.shape[1] * 0.1  # Max 10% of image
                
                best_mask_idx = -1
                best_score = 0
                
                for mask_idx in range(len(masks)):
                    mask_size = np.sum(masks[mask_idx])
                    if scores[mask_idx] > 0.5 and mask_size < max_size:
                        if scores[mask_idx] > best_score:
                            best_score = scores[mask_idx]
                            best_mask_idx = mask_idx
                
                if best_mask_idx >= 0:
                    mask_bool = masks[best_mask_idx].astype(bool)
                    
                    # Only assign pixels that aren't already part of another mask
                    unassigned = (composite_mask == 0) & mask_bool
                    
                    # Only create a new label if we have a reasonable number of pixels
                    if np.sum(unassigned) > 10:  # At least 10 pixels
                        composite_mask[unassigned] = label_id
                        label_id += 1
            except Exception as e:
                print(f"    ! Error on prompt {i}: {e}")
                continue

        final_mask = self.downscale_mask(composite_mask, original_shape)
        return final_mask, (label_id - 1)


# =====================
# UTILITIES
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

def visualize_sam2_zstack(img_zyx: np.ndarray, masks_3d: np.ndarray, centroids_per_slice, tracks, save_path: str, sample_interval: int = 5):
    """Visualize SAM2 results matching MEDIAR visualization style"""
    n_slices = img_zyx.shape[0]
    # Sample every Nth slice, up to max of 10 samples for visualization
    show_idx = np.arange(0, n_slices, sample_interval)
    if len(show_idx) > 10:
        # If still too many, subsample further
        show_idx = show_idx[::max(1, len(show_idx) // 10)]
    show_idx = show_idx[:10]  # Cap at 10 slices
    
    fig, axes = plt.subplots(3, len(show_idx), figsize=(3*len(show_idx), 9))
    if len(show_idx) == 1:
        axes = axes[:, np.newaxis]
    
    for i, z in enumerate(show_idx):
        # Row 0: Original image
        axes[0, i].imshow(img_zyx[z], cmap='gray')
        axes[0, i].set_title(f'Z={z}')
        axes[0, i].axis('off')
        
        # Row 1: Segmentation overlay with particle count
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
    parser = argparse.ArgumentParser(description="SAM 2 Upscaled Particle Tracking")
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to TIFF file or directory')
    parser.add_argument('--checkpoint', '-c', type=str, default = 'checkpoints/sam2_hiera_small.pt', help='Path to .pt checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Auto-detected if None.')
    parser.add_argument('--output', '-o', type=str, default='training_data_output/sam2', help='Output directory')
    parser.add_argument('--upscale', '-u', type=float, default=4.0, help='Upscale factor')
    parser.add_argument('--blob-threshold', type=float, default=0.05, help='Blob detection threshold (0.05 works well for TIFF)')
    parser.add_argument('--max-prompts', type=int, default=100, help='Maximum number of prompts to process per slice')
    parser.add_argument('--device', '-d', type=str, default='mps', help='Device (mps, cpu, cuda)')
    
    args = parser.parse_args()

    # Config Mapping
    if args.config is None:
        if "tiny" in args.checkpoint: args.config = "sam2_hiera_t.yaml"
        elif "small" in args.checkpoint: args.config = "sam2_hiera_s.yaml"
        elif "base" in args.checkpoint: args.config = "sam2_hiera_b+.yaml"
        elif "large" in args.checkpoint: args.config = "sam2_hiera_l.yaml"
        else:
            print("Could not auto-detect config from checkpoint name. Please use --config.")
            sys.exit(1)

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

    # Init Model
    try:
        predictor = SAM2PredictorUpscaled(
            checkpoint_path=args.checkpoint,
            model_cfg=args.config,
            device=args.device,
            upscale_factor=args.upscale,
            blob_threshold=args.blob_threshold
        )
    except Exception as e:
        print(f"Failed to load SAM 2: {e}")
        sys.exit(1)

    # Process
    for p in paths:
        print(f"\nProcessing {Path(p).name}...")
        try:
            stack = load_zstack(p)
        except Exception as e:
            print(f"Could not load {p}: {e}")
            continue

        masks_3d = np.zeros_like(stack, dtype=np.uint16)
        
        # Explicit verbose loop for debugging
        print(f"Starting segmentation on {stack.shape[0]} slices...")
        
        for z in range(stack.shape[0]):
            # Print status every slice to debug hanging
            print(f"\r  > Slice {z+1}/{stack.shape[0]}", end="", flush=True)
            
            masks_3d[z], count = predictor.predict(stack[z], max_prompts=args.max_prompts)

        print(f"\n  > Linking particles...", end="", flush=True)
        centroids_per_slice, counts = get_particles_per_slice(masks_3d)
        tracks = link_particles_across_slices(centroids_per_slice)
        tracks = filter_short_tracks(tracks)
        print(f" Done. Found {len(tracks)} tracks.")
        
        # Save
        base_name = Path(p).stem
        out_sub = Path(args.output) / base_name
        out_sub.mkdir(parents=True, exist_ok=True)
        
        # Save masks
        mask_path = out_sub / f"{base_name}_sam2_masks.tif"
        tifffile.imwrite(str(mask_path), masks_3d.astype(np.uint16))
        
        # Save visualization
        viz_path = out_sub / f"{base_name}_sam2_viz.png"
        visualize_sam2_zstack(stack, masks_3d, centroids_per_slice, tracks, 
                             save_path=str(viz_path), sample_interval=5)
        
        print(f"  > Results saved to {out_sub}")

if __name__ == "__main__":
    main()