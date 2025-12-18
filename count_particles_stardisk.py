"""
StarDist-based Nanoparticle Quantification
"""

import os
import sys
import argparse
import numpy as np
import tifffile
from pathlib import Path
from typing import Tuple, List, Dict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import scipy.optimize as opt
from scipy.stats import gaussian_kde
from skimage.filters import difference_of_gaussians

# StarDist Imports
try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
except ImportError:
    print("ERROR: StarDist not found. Please install: pip install stardist")
    sys.exit(1)

warnings.filterwarnings('ignore')


class StarDistQuantifier:
    def __init__(self, prob_thresh=0.5, nms_thresh=0.3, use_dog=False):
        print("Loading StarDist '2D_versatile_fluo' model...")
        self.model = StarDist2D.from_pretrained('2D_versatile_fluo')
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.use_dog = use_dog

    def predict_slice(self, image_slice: np.ndarray) -> np.ndarray:
        # Pre-process: Difference of Gaussians (DoG) to remove out-of-focus noise
        if self.use_dog:
            # low_sigma=1 preserves the core particle
            # high_sigma=4 removes the out-of-focus blur
            image_to_process = difference_of_gaussians(image_slice, low_sigma=1, high_sigma=4)
        else:
            image_to_process = image_slice

        # Normalize (Crucial for StarDist)
        img_norm = normalize(image_to_process, 1, 99.8, axis=(0, 1))
        
        # Predict
        labels, _ = self.model.predict_instances(
            img_norm,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh,
        )
        return labels

# =====================
# TRACKING UTILITIES
# =====================

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
    active_tracks = {} 

    for z, current_centroids in enumerate(centroids_per_slice):
        if not active_tracks:
            for y, x in current_centroids:
                tracks[next_tid] = [(z, y, x)]
                active_tracks[next_tid] = (y, x)
                next_tid += 1
            continue

        if not current_centroids:
            active_tracks = {}
            continue

        prev_ids = list(active_tracks.keys())
        prev_pos = np.array(list(active_tracks.values()))
        curr_pos = np.array(current_centroids)

        dist_matrix = cdist(prev_pos, curr_pos)
        valid_mask = dist_matrix < max_distance
        
        matched_track_ids = set()
        matched_detection_indices = set()

        if valid_mask.any():
            cost_matrix = np.where(valid_mask, dist_matrix, 1e10)
            row_inds, col_inds = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_inds, col_inds):
                if dist_matrix[r, c] < max_distance:
                    tid = prev_ids[r]
                    y, x = curr_pos[c]
                    tracks[tid].append((z, float(y), float(x)))
                    active_tracks[tid] = (float(y), float(x))
                    matched_track_ids.add(tid)
                    matched_detection_indices.add(c)

        new_active_tracks = {}
        for tid in matched_track_ids:
            new_active_tracks[tid] = active_tracks[tid]
        
        for idx, (y, x) in enumerate(current_centroids):
            if idx not in matched_detection_indices:
                tracks[next_tid] = [(z, float(y), float(x))]
                new_active_tracks[next_tid] = (float(y), float(x))
                next_tid += 1
        
        active_tracks = new_active_tracks

    return tracks

def filter_short_tracks(tracks: Dict[int, List], min_length: int = 3):
    return {tid: tr for tid, tr in tracks.items() if len(tr) >= min_length}

def filter_masks_by_size(masks_3d: np.ndarray, min_area: int = 0, max_area: int = None) -> np.ndarray:
    """
    Filter segmentation masks by particle area (size in pixels)
    
    Args:
        masks_3d: 3D array of segmentation masks
        min_area: Minimum particle area in pixels (default: 0, no minimum)
        max_area: Maximum particle area in pixels (default: None, no maximum)
    
    Returns:
        Filtered masks with only particles within size range
    """
    filtered_masks = np.zeros_like(masks_3d)
    
    for z in range(masks_3d.shape[0]):
        labels = masks_3d[z]
        props = measure.regionprops(labels.astype(np.int32))
        
        # Create new labels only for particles within size range
        new_label_id = 1
        for prop in props:
            area = prop.area
            if area >= min_area and (max_area is None or area <= max_area):
                # Copy this particle to filtered mask
                mask = labels == prop.label
                filtered_masks[z][mask] = new_label_id
                new_label_id += 1
    
    return filtered_masks
def analyze_track_brightness(stack, tracks, fov=6):
    print(f"\n--- Running Brightness Cluster Analysis on {len(tracks)} tracks ---")
    
    brightness_values = []
    
    # 1. Measure Brightness of every track
    for tid, positions in tracks.items():
        # Get the average position of this track
        ys = [p[1] for p in positions]
        xs = [p[2] for p in positions]
        mean_y, mean_x = int(np.mean(ys)), int(np.mean(xs))
        
        # Define crop boundaries
        y1, y2 = max(0, mean_y - fov), min(stack.shape[1], mean_y + fov)
        x1, x2 = max(0, mean_x - fov), min(stack.shape[2], mean_x + fov)
        
        # Extract the "tube" of the particle through the Z-stack
        # We assume the particle exists in the slices defined in 'positions'
        z_start = min([p[0] for p in positions])
        z_end = max([p[0] for p in positions]) + 1
        
        particle_crop = stack[z_start:z_end, y1:y2, x1:x2]
        
        if particle_crop.size == 0: continue

        # Create Average Intensity Projection (Average brightness over its lifetime)
        avg_img = np.mean(particle_crop, axis=0)
        
        # Fit Gaussian to get total brightness
        b_val, _ = integrateToGaussian(avg_img)
        
        if b_val is not None and b_val > 0 and not np.isnan(b_val):
            brightness_values.append(b_val)

    # 2. Cluster Analysis (Finding the "Single Unit")
    brightness_values = np.array(brightness_values)
    
    if len(brightness_values) < 5:
        print("Not enough tracks for cluster analysis.")
        return len(tracks)

    # Kernel Density Estimation to find the peak (Mode)
    density = gaussian_kde(brightness_values)
    xs = np.linspace(min(brightness_values), max(brightness_values), 200)
    ys = density(xs)
    
    # The mode of the distribution = Brightness of 1 Single Particle
    unit_brightness = xs[np.argmax(ys)]
    
    # Calculate how many particles are in each track
    # e.g. Track Brightness 200 / Unit 100 = 2.0 particles
    counts_per_track = brightness_values / unit_brightness
    estimated_total = np.sum(np.round(counts_per_track)) # Round to nearest integer

    print(f"Single Particle Unit Brightness: {unit_brightness:.2f}")
    print(f"StarDist Raw Tracks: {len(tracks)}")
    print(f"Cluster-Corrected Particle Count: {int(estimated_total)}")
    
    # Optional: Plot
    try:
        plt.figure(figsize=(6,4))
        plt.hist(brightness_values, bins=15, density=True, alpha=0.5, label='Track Brightness')
        plt.plot(xs, ys, 'r-', label='Density Fit')
        plt.axvline(unit_brightness, color='b', linestyle='--', label='Single Unit')
        plt.title(f"Cluster Analysis (Unit={unit_brightness:.0f})")
        plt.xlabel("Integrated Brightness")
        plt.legend()
        plt.savefig("brightness_clustering.png")
        plt.close()
    except Exception:
        pass

    return int(estimated_total)

# =====================
# BRIGHTNESS & GAUSSIAN FITTING HELPERS
# =====================

def gaussian(height, center_x, center_y, width_x, width_y, bk):
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2) + bk

def moments(data):
    """Returns (height, x, y, width_x, width_y, background)"""
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    bk = data.min()
    return height, x, y, width_x, width_y, bk

def fitgaussian(data):
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, hess_inv, _, _, _ = opt.leastsq(errorfunction, params, full_output=1)
    return p

def integrateToGaussian(data):
    """Fits Gaussian and calculates total integrated brightness"""
    try:
        popt = fitgaussian(data)
        # popt = [height, center_x, center_y, width_x, width_y, bg]
        # Volume of 2D Gaussian = 2 * pi * A * sigma_x * sigma_y
        # We subtract background for the integration
        height = popt[0]
        wx = np.abs(popt[3])
        wy = np.abs(popt[4])
        
        integrated_value = 2 * np.pi * height * wx * wy
        return integrated_value, popt
    except Exception:
        return None, None
    
# =====================
# VISUALIZATION
# =====================

def visualize_results(stack, masks, tracks, out_dir, base_name, sample_interval=5):
    """
    Visualize results, sampling every N slices (matching MEDIAR format)
    
    Creates a 3-row gallery:
    - Row 1: Raw grayscale images
    - Row 2: Segmentation overlay
    - Row 3: Tracked particles overlay
    """
    n_slices = stack.shape[0]
    
    # Sample every Nth slice, up to max of 10 samples for visualization
    show_idx = np.arange(0, n_slices, sample_interval)
    if len(show_idx) > 10:
        # If still too many, subsample further
        show_idx = show_idx[::max(1, len(show_idx) // 10)]
    show_idx = show_idx[:10]  # Cap at 10 slices
    
    fig, axes = plt.subplots(3, len(show_idx), figsize=(3*len(show_idx), 9))
    if len(show_idx) == 1:
        axes = axes[:, np.newaxis]
    
    # Count detections per slice for titles
    centroids_per_slice = []
    for z in range(n_slices):
        cents = get_centroids_from_labels(masks[z])
        centroids_per_slice.append(cents)
    
    for i, z in enumerate(show_idx):
        # Row 1: Raw image
        axes[0, i].imshow(stack[z], cmap='gray')
        axes[0, i].set_title(f'Z={z}')
        axes[0, i].axis('off')
        
        # Row 2: Segmentation overlay
        axes[1, i].imshow(stack[z], cmap='gray', alpha=0.6)
        axes[1, i].imshow(masks[z], cmap='tab20', alpha=0.4, vmin=0, vmax=20)
        axes[1, i].set_title(f'Detected: {len(centroids_per_slice[z])}')
        axes[1, i].axis('off')
        
        # Row 3: Tracked particles overlay
        axes[2, i].imshow(stack[z], cmap='gray', alpha=0.5)
        axes[2, i].imshow(masks[z], cmap='tab20', alpha=0.5, vmin=0, vmax=20)
        for tid, positions in tracks.items():
            for pz, py, px in positions:
                if pz == z:
                    axes[2, i].plot(px, py, 'r.', markersize=3)
        axes[2, i].set_title('Tracked')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    gal_path = out_dir / f"{base_name}_stardist_viz.png"
    plt.savefig(gal_path, dpi=150, bbox_inches='tight')
    plt.close()

    return gal_path

def main():
    parser = argparse.ArgumentParser(description="StarDist Nanoparticle Counter v3")
    parser.add_argument('--image', '-i', type=str, required=True, help='Path to TIFF Z-stack')
    parser.add_argument('--output', '-o', type=str, default='training_data_output/stardist', help='Output directory')
    parser.add_argument('--viz-sample-interval', type=int, default=10, 
                        help='Sample every Nth slice for visualization (default: 5)')
    
    # Params
    parser.add_argument('--prob_thresh', type=float, default=0.6, help='StarDist Probability (higher = less sensitive, range: 0-1)')
    parser.add_argument('--nms_thresh', type=float, default=0.3, help='Overlap Threshold')
    parser.add_argument('--max_dist', type=float, default=10.0, help='Max jump dist')
    parser.add_argument('--min_length', type=int, default=3, help='Min track length')
    parser.add_argument('--min_area', type=int, default=0, help='Minimum particle area in pixels (default: 0, no filter)')
    parser.add_argument('--max_area', type=int, default=None, help='Maximum particle area in pixels (default: None, no filter)')
    parser.add_argument('--use_dog', action='store_true', 
                        help='Apply Difference of Gaussians filter to split clumped/out-of-focus particles.')

    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Load
    try:
        stack = tifffile.imread(args.image)
        if stack.ndim != 3:
            print(f"Error: Image is {stack.ndim}D. Expected 3D (Z,Y,X).")
            return
    except Exception as e:
        print(f"Load Error: {e}")
        return

    # Run
    quantifier = StarDistQuantifier(prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh, use_dog=args.use_dog)
    masks_3d = np.zeros(stack.shape, dtype=np.uint16)
    
    print(f"Segmenting {stack.shape[0]} slices...")
    for z in tqdm(range(stack.shape[0])):
        masks_3d[z] = quantifier.predict_slice(stack[z])

    # Filter by size if specified
    if args.min_area > 0 or args.max_area is not None:
        print(f"Filtering particles by size (min: {args.min_area}, max: {args.max_area})...")
        masks_3d = filter_masks_by_size(masks_3d, min_area=args.min_area, max_area=args.max_area)

    # Track
    centroids, counts = get_particles_per_slice(masks_3d)
    tracks = link_particles_across_slices(centroids, max_distance=args.max_dist)
    final_tracks = filter_short_tracks(tracks, min_length=args.min_length)

    print(f"\nRESULTS: {len(final_tracks)} unique particles tracked.")

    # Save
    base_name = Path(args.image).stem
    tifffile.imwrite(out_dir / f"{base_name}_masks.tif", masks_3d)
    corrected_count = analyze_track_brightness(stack, final_tracks)
    gal_path = visualize_results(stack, masks_3d, final_tracks, out_dir, base_name, 
                                 sample_interval=args.viz_sample_interval)
    print(f"Saved Visualization: {gal_path}")

if __name__ == "__main__":
    main()