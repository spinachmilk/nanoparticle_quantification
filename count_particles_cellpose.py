"""
Cellpose-based Nanoparticle Segmentation with Image Upscaling 

Key features:
- Input: 256x256 with ~15-pixel diameter particles
- Upscale by 4x -> 1024x1024 with ~60-pixel particles
- Process with Cellpose
- Downscale results back to original resolution
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tifffile
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check if Cellpose is installed
try:
    from cellpose import models
except ImportError as e:
    print("ERROR: Cellpose not installed.")
    print("pip install cellpose")
    sys.exit(1)


class CellposePredictorUpscaled:
    """Cellpose Predictor with automatic upscaling for small particles"""

    def __init__(
        self,
        model_type: str = "cyto2",
        device: str = "cpu",
        diameter: float = 30.0,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        upscale_factor: float = 4.0,
    ):
        """
        Initialize Cellpose predictor with upscaling
        
        Args:
            model_type: Cellpose model type (cyto, cyto2, nuclei, etc.)
            device: 'cpu' or 'cuda'
            diameter: Expected diameter of particles (in original pixels)
            flow_threshold: Cellpose flow threshold
            cellprob_threshold: Cellpose probability threshold
            upscale_factor: Factor to upscale images (e.g., 4.0 = 4x larger)
        """
        self.use_gpu = device == "cuda"
        self.upscale_factor = upscale_factor
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

        print(f"Initializing Cellpose ({model_type}) with {upscale_factor}x upscaling...")
        # Cellpose loads the model into memory here
        self.model = models.CellposeModel(gpu=self.use_gpu, model_type=model_type)

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using bicubic interpolation"""
        if self.upscale_factor == 1.0:
            return image
        
        h, w = image.shape[:2]
        new_h = int(h * self.upscale_factor)
        new_w = int(w * self.upscale_factor)
        
        # cv2.resize works for 2D arrays (H,W) or (H,W,C)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return upscaled

    def downscale_mask(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Downscale segmentation mask back to original size using nearest neighbor"""
        if self.upscale_factor == 1.0:
            return mask
        
        # Use nearest neighbor to preserve label IDs (integers)
        # Note: cv2.resize expects (width, height)
        downscaled = cv2.resize(
            mask.astype(np.float32), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        return downscaled.astype(mask.dtype)

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, None, None]:
        """
        Run inference with upscaling
        
        Args:
            image: Input image (original resolution)
            
        Returns:
            Tuple of (instance_mask, flow_field, cell_probability)
            Note: Flow and Prob are returned as None to match MEDIAR signature, 
            unless you strictly need them for debug.
        """
        original_shape = image.shape[:2]
        
        # 1. Upscale input
        upscaled_image = self.upscale_image(image)
        
        # Calculate scaled diameter
        scaled_diameter = self.diameter * self.upscale_factor if self.diameter > 0 else None
        
        # 2. Run Cellpose
        # channels=[0,0] means grayscale
        masks, flows, styles = self.model.eval(
            upscaled_image, 
            diameter=scaled_diameter,
            channels=[0,0],
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold
        )
        
        # 3. Downscale back to original resolution
        instance_mask = self.downscale_mask(masks, original_shape)
        
        # We return None for flow/prob to match the MEDIAR script's unpacking signature
        return instance_mask, None, None


# =====================
# Z-STACK UTILITIES
# (Identical to MEDIAR Script)
# =====================

def load_zstack(image_path: str) -> np.ndarray:
    """Load a z-stack TIFF file"""
    img = tifffile.imread(image_path)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    return img


def get_centroids_from_labels(labels: np.ndarray) -> List[Tuple[float, float]]:
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {labels.shape}")
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


def link_particles_across_slices(centroids_per_slice: List[List[Tuple[float, float]]], max_distance: float = 10.0):
    tracks = {}
    next_tid = 1
    active = {}
    for z, cents in enumerate(centroids_per_slice):
        if z == 0:
            for y, x in cents:
                tracks[next_tid] = [(z, y, x)]
                active[next_tid] = (y, x)
                next_tid += 1
            continue
        if not active or not cents:
            for y, x in cents:
                tracks[next_tid] = [(z, y, x)]
                active[next_tid] = (y, x)
                next_tid += 1
            continue
        prev_pos = np.array(list(active.values()))
        curr_pos = np.array(cents)
        D = cdist(prev_pos, curr_pos)
        valid = D < max_distance
        matched_tracks = set()
        matched_dets = set()
        if valid.any():
            row_ind, col_ind = linear_sum_assignment(np.where(valid, D, 1e10))
            tids = list(active.keys())
            for r, c in zip(row_ind, col_ind):
                if D[r, c] < max_distance:
                    tid = tids[r]
                    y, x = curr_pos[c]
                    tracks[tid].append((z, float(y), float(x)))
                    active[tid] = (float(y), float(x))
                    matched_tracks.add(tid)
                    matched_dets.add(c)
        for tid in list(active.keys()):
            if tid not in matched_tracks:
                del active[tid]
        for idx, (y, x) in enumerate(cents):
            if idx not in matched_dets:
                tracks[next_tid] = [(z, float(y), float(x))]
                active[next_tid] = (float(y), float(x))
                next_tid += 1
    return tracks


def filter_short_tracks(tracks: Dict[int, List[Tuple[int, float, float]]], min_length: int = 3):
    return {tid: tr for tid, tr in tracks.items() if len(tr) >= min_length}


def segment_zstack_cellpose(img_zyx: np.ndarray, predictor: CellposePredictorUpscaled) -> np.ndarray:
    """Segment each z-slice using Cellpose predictor"""
    Z = img_zyx.shape[0]
    masks_3d = np.zeros(img_zyx.shape, dtype=np.uint16)
    
    # We use the predictor's upscale factor in the description
    print(f"Segmenting {Z} slices (Upscale factor: {predictor.upscale_factor}x)...")
    
    for z in tqdm(range(Z), desc="Cellpose Inference"):
        slice2d = img_zyx[z]
        inst_mask, _, _ = predictor.predict(slice2d)
        masks_3d[z] = inst_mask.astype(np.uint16)
    return masks_3d


def visualize_zstack_results(img_zyx: np.ndarray, masks_3d: np.ndarray, centroids_per_slice, tracks, save_path: str, sample_interval: int = 5):
    """Visualize results, sampling every N slices (Same style as MEDIAR)"""
    n_slices = img_zyx.shape[0]
    # Sample every Nth slice, up to max of 10 samples for visualization
    show_idx = np.arange(0, n_slices, sample_interval)
    if len(show_idx) > 10:
        # If still too many, subsample further
        show_idx = show_idx[::max(1, len(show_idx) // 10)]
    show_idx = show_idx[:10]  # Cap at 10 slices
    
    if len(show_idx) == 0:
        return

    fig, axes = plt.subplots(3, len(show_idx), figsize=(3*len(show_idx), 9))
    
    # Handle case where there is only 1 slice to show (axes is 1D)
    if len(show_idx) == 1:
        axes = axes[:, np.newaxis]
        
    for i, z in enumerate(show_idx):
        # Row 0: Raw Image
        axes[0, i].imshow(img_zyx[z], cmap='gray')
        axes[0, i].set_title(f'Z={z}')
        axes[0, i].axis('off')
        
        # Row 1: Segmentation Overlay
        axes[1, i].imshow(img_zyx[z], cmap='gray', alpha=0.6)
        axes[1, i].imshow(masks_3d[z], cmap='tab20', alpha=0.4, vmin=0, vmax=20)
        axes[1, i].set_title(f'Detected: {len(centroids_per_slice[z])}')
        axes[1, i].axis('off')
        
        # Row 2: Tracking Overlay
        axes[2, i].imshow(img_zyx[z], cmap='gray', alpha=0.5)
        # Optional: Show mask faintly in background
        axes[2, i].imshow(masks_3d[z], cmap='tab20', alpha=0.2, vmin=0, vmax=20)
        
        # Plot Tracks
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
    parser = argparse.ArgumentParser(
        description="Cellpose with upscaling for nanoparticle segmentation (Matches MEDIAR pipeline)"
    )
    parser.add_argument('--image', '-i', type=str, 
                        required=True,
                        help='Path to TIFF file or directory')
    parser.add_argument('--output', '-o', type=str, 
                        default='cellpose_output',
                        help='Output directory')
    # Cellpose Arguments
    parser.add_argument('--model_type', '-m', type=str, default='cyto2',
                        help='Cellpose model type (cyto, cyto2, nuclei)')
    parser.add_argument('--diameter', type=float, default=15,
                        help='Diameter of particles (in original pixels, before upscaling)')
    parser.add_argument('--flow_threshold', type=float, default=0.4)
    parser.add_argument('--cellprob_threshold', type=float, default=0.0)
    
    # Pipeline Arguments
    parser.add_argument('--device', '-d', type=str, default='cpu', 
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for inference')
    parser.add_argument('--upscale-factor', '-u', type=float, default=4.0,
                        help='Image upscaling factor (e.g., 4.0 = 4x larger)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--max-link-distance', type=float, default=10.0,
                        help='Max distance to link particles between slices')
    parser.add_argument('--min-track-length', type=int, default=3,
                        help='Minimum track length')
    parser.add_argument('--viz-sample-interval', type=int, default=5,
                        help='Sample every Nth slice for visualization')

    args = parser.parse_args()

    # Resolve inputs
    if os.path.isfile(args.image):
        paths = [args.image]
    elif os.path.isdir(args.image):
        all_files = sorted(os.listdir(args.image))
        paths = [str(Path(args.image) / f) for f in all_files
                 if Path(f).suffix.lower() in ['.tif', '.tiff'] and not f.startswith('.')]
        if not paths:
            print('No TIFF files found.')
            return
        print(f"Found {len(paths)} TIFF files")
    else:
        raise ValueError(f"Invalid image path: {args.image}")

    os.makedirs(args.output, exist_ok=True)

    # Initialize predictor
    predictor = CellposePredictorUpscaled(
        model_type=args.model_type,
        device=args.device,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        upscale_factor=args.upscale_factor,
    )

    summary = []
    for p in paths:
        img_name = Path(p).name
        out_dir = Path(args.output) / Path(p).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing: {img_name}")

        try:
            stack = load_zstack(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

        if stack.ndim == 3:
            # 1. Segment (with upscale logic)
            masks_3d = segment_zstack_cellpose(stack, predictor)
            
            # 2. Get Centroids
            centroids_per_slice, counts = get_particles_per_slice(masks_3d)
            
            # 3. Track (Same Hungarian Algorithm as MEDIAR)
            tracks = link_particles_across_slices(centroids_per_slice, max_distance=args.max_link_distance)
            tracks = filter_short_tracks(tracks, min_length=args.min_track_length)

            # 4. Save Results
            mask_path = out_dir / f"{Path(p).stem}_cellpose_upscaled_masks.tif"
            tifffile.imwrite(str(mask_path), masks_3d.astype(np.uint16))
            
            if not args.no_viz:
                viz_path = out_dir / f"{Path(p).stem}_cellpose_upscaled_viz.png"
                visualize_zstack_results(stack, masks_3d, centroids_per_slice, tracks, 
                                       save_path=str(viz_path), sample_interval=args.viz_sample_interval)

            n_tracks = len(tracks)
            summary.append((img_name, n_tracks))
            print(f"Detected {n_tracks} particle tracks")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, count in summary:
        print(f"{name}: {count} tracks")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()