import numpy as np
import tifffile
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage import measure, filters, feature
from scipy import ndimage


def load_zstack(image_path):
    """
    Load a z-stack TIFF file
    """
    img = tifffile.imread(image_path)
    print(f"Loaded z-stack: {img.shape}")
    if len(img.shape) == 2:
        img = img[np.newaxis, ...]
    print(f"Shape (Z, Y, X): {img.shape}")
    print(f"Data type: {img.dtype}")
    print(f"Value range: [{img.min()}, {img.max()}]")
    return img


def normalize_for_sam(img):
    """
    Normalize image to 0-255 uint8 for SAM 2
    """
    img_norm = img.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
    img_norm = (img_norm * 255).astype(np.uint8)
    return img_norm


def convert_to_rgb(img_gray):
    """
    Convert grayscale to RGB by repeating channels
    """
    return np.stack([img_gray, img_gray, img_gray], axis=-1)


def generate_point_prompts(slice_img, method='blob_detection', 
                          min_sigma=1, max_sigma=5, num_sigma=10,
                          threshold=0.01, overlap=0.5):
    """
    Automatically generate point prompts for SAM 2 using blob detection
    
    Parameters:
    -----------
    slice_img : 2D array
        Input image slice
    method : str
        'blob_detection' for LoG blob detection, 'peak_detection' for local maxima
    min_sigma, max_sigma : float
        Range of blob sizes to detect
    num_sigma : int
        Number of scales to search
    threshold : float
        Blob detection threshold (lower = more blobs)
    overlap : float
        Maximum overlap between blobs (0-1)
    
    Returns:
    --------
    points : array (N, 2)
        Point coordinates [x, y]
    """
    if method == 'blob_detection':
        from skimage.feature import blob_log
        
        # Normalize image
        img_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
        
        # Detect blobs
        blobs = blob_log(
            img_norm,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap
        )
        
        if len(blobs) > 0:
            # Extract (y, x) positions
            points = blobs[:, :2]  # (y, x)
            # Swap to (x, y) for SAM
            points = points[:, [1, 0]]
            return points
        else:
            return np.array([]).reshape(0, 2)
    
    elif method == 'peak_detection':
        # Use intensity peaks
        from skimage.feature import peak_local_max
        
        # Smooth slightly
        smoothed = ndimage.gaussian_filter(slice_img, sigma=1.0)
        
        # Find peaks
        peaks = peak_local_max(
            smoothed,
            min_distance=5,
            threshold_abs=np.percentile(slice_img, 70),  # Top 30% intensity
            exclude_border=True
        )
        
        if len(peaks) > 0:
            # Swap to (x, y)
            points = peaks[:, [1, 0]]
            return points
        else:
            return np.array([]).reshape(0, 2)
    
    elif method == 'adaptive_threshold':
        # Use adaptive thresholding to find candidates
        threshold = filters.threshold_local(slice_img, block_size=21, offset=-10)
        binary = slice_img > threshold
        
        # Label regions
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        
        # Get centroids
        if len(props) > 0:
            points = np.array([[prop.centroid[1], prop.centroid[0]] for prop in props])
            return points
        else:
            return np.array([]).reshape(0, 2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def segment_with_sam2(img, model, predictor, 
                      prompt_method='blob_detection',
                      blob_threshold=0.01,
                      min_sigma=1, max_sigma=5):
    """
    Segment z-stack using SAM 2 with automatic prompt generation
    
    Parameters:
    -----------
    img : 3D array (Z, Y, X)
        Input z-stack
    model : SAM2VideoPredictor
        SAM 2 model
    predictor : SAM2ImagePredictor
        SAM 2 image predictor
    prompt_method : str
        Method for generating prompts ('blob_detection', 'peak_detection', 'adaptive_threshold')
    """
    n_slices = img.shape[0]
    masks_3d = np.zeros_like(img, dtype=np.uint16)
    centroids_per_slice = []
    particles_per_slice = []
    
    print(f"\nSegmenting {n_slices} slices with SAM 2...")
    print(f"Prompt generation method: {prompt_method}")
    
    for z in range(n_slices):
        slice_img = img[z]
        
        # Normalize and convert to RGB
        slice_norm = normalize_for_sam(slice_img)
        slice_rgb = convert_to_rgb(slice_norm)
        
        # Generate point prompts
        points = generate_point_prompts(
            slice_img,
            method=prompt_method,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=blob_threshold
        )
        
        if len(points) == 0:
            centroids_per_slice.append([])
            particles_per_slice.append(0)
            if (z + 1) % 10 == 0 or z == n_slices - 1:
                print(f"  Slice {z+1}/{n_slices}: 0 particles detected (no prompts)")
            continue
        
        # Set image for SAM
        predictor.set_image(slice_rgb)
        
        # Segment each prompt point
        slice_masks = np.zeros(slice_img.shape, dtype=np.uint16)
        label_id = 1
        centroids = []
        
        for point in points:
            try:
                # Predict mask for this point
                masks, scores, logits = predictor.predict(
                    point_coords=point.reshape(1, 2),
                    point_labels=np.array([1]),  # 1 = foreground
                    multimask_output=False
                )
                
                # Use the mask with highest score
                if len(masks) > 0 and scores[0] > 0.5:
                    mask = masks[0]
                    
                    # Add to slice masks
                    slice_masks[mask] = label_id
                    
                    # Get centroid
                    y_coords, x_coords = np.where(mask)
                    if len(y_coords) > 0:
                        centroid = (y_coords.mean(), x_coords.mean())
                        centroids.append(centroid)
                        label_id += 1
            
            except Exception as e:
                # Skip problematic points
                continue
        
        masks_3d[z] = slice_masks
        centroids_per_slice.append(centroids)
        particles_per_slice.append(len(centroids))
        
        if (z + 1) % 10 == 0 or z == n_slices - 1:
            print(f"  Slice {z+1}/{n_slices}: {len(centroids)} particles detected (from {len(points)} prompts)")
    
    print(f"\nTotal detections across all slices: {sum(particles_per_slice)}")
    return masks_3d, centroids_per_slice


def link_particles_across_slices(centroids_per_slice, max_distance=10):
    """
    Link particles across consecutive z-slices using Hungarian algorithm
    """
    print(f"\nLinking particles across slices (max distance: {max_distance} px)...")
    
    tracks = {}
    next_track_id = 1
    active_tracks = {}
    
    for z in range(len(centroids_per_slice)):
        current_centroids = centroids_per_slice[z]
        
        if z == 0:
            for y, x in current_centroids:
                tracks[next_track_id] = [(z, y, x)]
                active_tracks[next_track_id] = (y, x)
                next_track_id += 1
        else:
            if len(active_tracks) == 0 or len(current_centroids) == 0:
                for y, x in current_centroids:
                    tracks[next_track_id] = [(z, y, x)]
                    active_tracks[next_track_id] = (y, x)
                    next_track_id += 1
            else:
                prev_positions = np.array(list(active_tracks.values()))
                curr_positions = np.array(current_centroids)
                distance_matrix = cdist(prev_positions, curr_positions)
                
                track_ids = list(active_tracks.keys())
                matched_tracks = set()
                matched_detections = set()
                
                valid_matches = distance_matrix < max_distance
                
                if valid_matches.any():
                    row_ind, col_ind = linear_sum_assignment(
                        np.where(valid_matches, distance_matrix, 1e10)
                    )
                    
                    for track_idx, det_idx in zip(row_ind, col_ind):
                        if distance_matrix[track_idx, det_idx] < max_distance:
                            track_id = track_ids[track_idx]
                            y, x = current_centroids[det_idx]
                            tracks[track_id].append((z, y, x))
                            active_tracks[track_id] = (y, x)
                            matched_tracks.add(track_id)
                            matched_detections.add(det_idx)
                
                for track_id in track_ids:
                    if track_id not in matched_tracks:
                        del active_tracks[track_id]
                
                for det_idx, (y, x) in enumerate(current_centroids):
                    if det_idx not in matched_detections:
                        tracks[next_track_id] = [(z, y, x)]
                        active_tracks[next_track_id] = (y, x)
                        next_track_id += 1
    
    print(f"Linked into {len(tracks)} unique particle tracks")
    return tracks


def filter_short_tracks(tracks, min_length=3):
    """
    Remove tracks that appear in too few slices
    """
    filtered = {tid: track for tid, track in tracks.items() if len(track) >= min_length}
    n_removed = len(tracks) - len(filtered)
    if n_removed > 0:
        print(f"Removed {n_removed} short tracks (< {min_length} slices)")
    return filtered


def get_track_properties(tracks):
    """
    Extract properties of each particle track
    """
    properties = {}
    
    for track_id, positions in tracks.items():
        z_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        x_coords = [p[2] for p in positions]
        
        properties[track_id] = {
            'n_slices': len(positions),
            'z_range': (min(z_coords), max(z_coords)),
            'z_extent': max(z_coords) - min(z_coords) + 1,
            'centroid': (np.mean(z_coords), np.mean(y_coords), np.mean(x_coords)),
            'trajectory': positions
        }
    
    return properties


def visualize_results(img, masks_3d, centroids_per_slice, tracks, output_path=None):
    """
    Visualize segmentation and tracking results
    """
    n_slices = img.shape[0]
    slices_to_show = np.linspace(0, n_slices-1, min(6, n_slices), dtype=int)
    
    fig, axes = plt.subplots(3, len(slices_to_show), figsize=(18, 9))
    if len(slices_to_show) == 1:
        axes = axes[:, np.newaxis]
    
    for i, z in enumerate(slices_to_show):
        # Original
        axes[0, i].imshow(img[z], cmap='gray')
        axes[0, i].set_title(f'Z={z}')
        axes[0, i].axis('off')
        
        # Segmented with labels
        axes[1, i].imshow(img[z], cmap='gray', alpha=0.6)
        axes[1, i].imshow(masks_3d[z], cmap='tab20', alpha=0.4, vmin=0, vmax=20)
        axes[1, i].set_title(f'Detected: {len(centroids_per_slice[z])}')
        axes[1, i].axis('off')
        
        # Tracked particles
        axes[2, i].imshow(img[z], cmap='gray', alpha=0.5)
        axes[2, i].imshow(masks_3d[z], cmap='tab20', alpha=0.5, vmin=0, vmax=20)
        
        for track_id, positions in tracks.items():
            for pz, py, px in positions:
                if pz == z:
                    axes[2, i].plot(px, py, 'r.', markersize=3)
        
        axes[2, i].set_title(f'Tracked')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")
    else:
        plt.show()
    
    return fig


def visualize_particle_count_changes(centroids_per_slice, output_path=None):
    """
    Plot the number of detected particles per slice
    """
    particle_counts = [len(centroids) for centroids in centroids_per_slice]
    z_indices = range(len(particle_counts))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(z_indices, particle_counts, 'b-o', markersize=4, linewidth=2)
    ax.set_xlabel('Z-Slice Index', fontsize=12)
    ax.set_ylabel('Number of Detected Particles', fontsize=12)
    ax.set_title('Particle Count per Z-Slice (SAM 2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    mean_count = np.mean(particle_counts)
    std_count = np.std(particle_counts)
    
    ax.text(0.02, 0.98, f'Mean: {mean_count:.1f} ± {std_count:.1f}\n' + 
                        f'Range: {min(particle_counts)} - {max(particle_counts)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved particle count plot to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    """
    Main workflow for SAM 2-based nanoparticle detection
    """
    
    # Input file
    IMAGE_PATH = "/Users/cindyli/nanoparticle-tracking/training_data_cropped/2025107_R8_1pBSA_1pCasein_cell5_163_cropped.tif"
    
    # SAM 2 model configuration
    MODEL_SIZE = 'small'  # Options: 'tiny', 'small', 'base_plus', 'large'
    USE_GPU = torch.cuda.is_available()
    
    # Prompt generation parameters
    PROMPT_METHOD = 'blob_detection'  # Options: 'blob_detection', 'peak_detection', 'adaptive_threshold'
    BLOB_THRESHOLD = 0.01  # Lower = more prompts (try 0.005-0.02)
    MIN_SIGMA = 1  # Minimum blob size
    MAX_SIGMA = 5  # Maximum blob size
    
    # Tracking parameters
    MAX_LINK_DISTANCE = 10  # Max pixel distance to link particles between slices
    MIN_TRACK_LENGTH = 3  # Minimum number of slices a particle must appear in
    
    # ====================================
    
    print("=" * 70)
    print("SAM 2-BASED NANOPARTICLE DETECTION")
    print("=" * 70)
    
    # Check for SAM 2
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("\n ERROR: SAM 2 not installed!")
        print("\nTo install SAM 2:")
        print("1. pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        print("2. Download model checkpoint:")
        print("   mkdir -p checkpoints")
        print("   cd checkpoints")
        print("   wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt")
        print("\nOr run the installation script I'll create...")
        return None, None
    
    # Load SAM 2 model
    print(f"\nLoading SAM 2 model (size: {MODEL_SIZE})...")
    print(f"GPU available: {USE_GPU}")
    
    try:
        checkpoint_map = {
            'tiny': 'checkpoints/sam2_hiera_tiny.pt',
            'small': 'checkpoints/sam2_hiera_small.pt',
            'base_plus': 'checkpoints/sam2_hiera_base_plus.pt',
            'large': 'checkpoints/sam2_hiera_large.pt'
        }
        
        config_map = {
            'tiny': 'sam2_hiera_t.yaml',
            'small': 'sam2_hiera_s.yaml',
            'base_plus': 'sam2_hiera_b+.yaml',
            'large': 'sam2_hiera_l.yaml'
        }
        
        checkpoint = checkpoint_map[MODEL_SIZE]
        config = config_map[MODEL_SIZE]
        
        model = build_sam2(config, checkpoint, device='cuda' if USE_GPU else 'cpu')
        predictor = SAM2ImagePredictor(model)
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"\n Error loading model: {e}")
        print("\nMake sure you have downloaded the SAM 2 checkpoint.")
        return None, None
    
    image_path = Path(IMAGE_PATH)
    
    # Load image
    print(f"\nLoading: {image_path.name}")
    img = load_zstack(IMAGE_PATH)
    
    # Segment with SAM 2
    masks_3d, centroids_per_slice = segment_with_sam2(
        img,
        model=model,
        predictor=predictor,
        prompt_method=PROMPT_METHOD,
        blob_threshold=BLOB_THRESHOLD,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA
    )
    
    # Link particles across slices
    tracks = link_particles_across_slices(
        centroids_per_slice,
        max_distance=MAX_LINK_DISTANCE
    )
    
    # Filter short tracks
    tracks = filter_short_tracks(tracks, min_length=MIN_TRACK_LENGTH)
    
    # Get properties
    properties = get_track_properties(tracks)
    
    # Results
    n_particles = len(tracks)
    print("\n" + "=" * 70)
    print(f"TOTAL UNIQUE PARTICLES: {n_particles}")
    print("=" * 70)
    
    # Summary statistics
    if properties:
        z_extents = [p['z_extent'] for p in properties.values()]
        n_slices_list = [p['n_slices'] for p in properties.values()]
        
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Detection method: SAM 2 ({MODEL_SIZE})")
        print(f"Prompt method: {PROMPT_METHOD}")
        print(f"Total unique particles: {n_particles}")
        print(f"Average track length: {np.mean(n_slices_list):.1f} ± {np.std(n_slices_list):.1f} slices")
        print(f"Average z-extent: {np.mean(z_extents):.1f} ± {np.std(z_extents):.1f} slices")
        print(f"Track length range: {min(n_slices_list)} - {max(n_slices_list)} slices")
        
        print("\nExample particle tracks:")
        print(f"  {'Track ID':<10} {'Slices':<10} {'Z-Range':<15}")
        print("  " + "-" * 40)
        for track_id, props in sorted(properties.items())[:10]:
            print(f"  {track_id:<10} {props['n_slices']:<10} {props['z_range'][0]}-{props['z_range'][1]}")
        if len(properties) > 10:
            print(f"  ... and {len(properties) - 10} more")
    
    # Save masks
    mask_path = image_path.parent / f"{image_path.stem}_sam2_masks.tif"
    tifffile.imwrite(mask_path, masks_3d.astype(np.uint16))
    print(f"\nSaved masks to: {mask_path}")
    
    # Visualizations
    viz_path = image_path.parent / f"{image_path.stem}_sam2_viz.png"
    visualize_results(img, masks_3d, centroids_per_slice, tracks, output_path=viz_path)
    
    count_path = image_path.parent / f"{image_path.stem}_sam2_counts.png"
    visualize_particle_count_changes(centroids_per_slice, output_path=count_path)
    
    return tracks, properties


if __name__ == "__main__":
    main()
