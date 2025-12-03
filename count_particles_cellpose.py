import numpy as np
import tifffile
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models, io
from skimage import measure
import warnings
warnings.filterwarnings('ignore')


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


def segment_with_cellpose(img, model_type='cyto3', diameter=None, channels=None,
                          flow_threshold=0.4, cellprob_threshold=0.0,
                          use_gpu=False, stitch_threshold=0.0):
    """
    Segment all slices using Cellpose
    
    Parameters:
    -----------
    img : 3D array (Z, Y, X)
        Input z-stack
    model_type : str
        Cellpose model: 'cyto', 'cyto2', 'cyto3', 'nuclei', or path to custom model
    diameter : float or None
        Expected object diameter in pixels. If None, auto-estimated
    channels : list or None
        [cytoplasm, nucleus] channels. Use None for grayscale z-stacks
    flow_threshold : float
        Flow error threshold (0-1). Lower = more cells. Default 0.4
    cellprob_threshold : float  
        Cell probability threshold (-6 to 6). Lower = more cells. Default 0.0
    use_gpu : bool
        Use GPU acceleration if available
    stitch_threshold : float
        Stitch 3D masks across slices (0 = no stitching)
    
    Returns:
    --------
    masks_3d : 3D array
        Labeled masks for each slice
    """
    print(f"\nSegmenting with Cellpose model: {model_type}")
    print(f"Diameter: {diameter if diameter else 'auto-estimate'}")
    print(f"Flow threshold: {flow_threshold}")
    print(f"Cell probability threshold: {cellprob_threshold}")
    print(f"GPU: {use_gpu}")
    
    # Load model
    model = models.CellposeModel(model_type=model_type, gpu=use_gpu)
    
    # Process each slice individually
    n_slices = img.shape[0]
    masks_3d = np.zeros_like(img, dtype=np.uint16)
    diams_list = []
    
    print(f"Processing {n_slices} slices...")
    
    for z in range(n_slices):
        # Get single 2D slice
        slice_2d = img[z]
        
        # Run segmentation on this slice
        mask_2d, flows, diam = model.eval(
            slice_2d,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        
        masks_3d[z] = mask_2d
        diams_list.append(diam)
        
        if (z + 1) % 10 == 0 or z == n_slices - 1:
            print(f"  Processed slice {z+1}/{n_slices}")
    
    # Get mean diameter
    mean_diam = np.mean(diams_list)
    print(f"\nEstimated diameter: {mean_diam:.1f} pixels")
    print(f"Total objects detected across all slices: {masks_3d.max()}")
    
    return masks_3d, mean_diam


def get_centroids_from_labels(labels):
    """
    Extract centroids from labeled image
    """
    # Ensure labels is 2D
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels array, got shape {labels.shape}")
    
    # Convert to integer type if needed
    labels = labels.astype(np.int32)
    
    props = measure.regionprops(labels)
    centroids = [(prop.centroid[0], prop.centroid[1]) for prop in props]
    return centroids


def get_particles_per_slice(masks_3d):
    """
    Count particles in each slice and get centroids
    """
    centroids_per_slice = []
    particles_per_slice = []
    
    for z in range(masks_3d.shape[0]):
        centroids = get_centroids_from_labels(masks_3d[z])
        centroids_per_slice.append(centroids)
        particles_per_slice.append(len(centroids))
        
        if (z + 1) % 10 == 0 or z == masks_3d.shape[0] - 1:
            print(f"  Slice {z+1}/{masks_3d.shape[0]}: {len(centroids)} particles")
    
    return centroids_per_slice, particles_per_slice


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
    ax.set_title('Particle Count per Z-Slice (Cellpose)', fontsize=14, fontweight='bold')
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
    Main workflow for Cellpose-based nanoparticle detection
    """
    # ========== CONFIGURATION ==========
    
    # Input file
    IMAGE_PATH = "/Users/cindyli/nanoparticle-tracking/training_data_cropped/2025107_R8_1pBSA_1pCasein_cell5_163_cropped.tif"
    
    # Cellpose parameters
    MODEL_TYPE = 'cyto3'  # Options: 'cyto', 'cyto2', 'cyto3', 'nuclei', or path to custom model
    DIAMETER = 15  # Expected particle diameter in pixels (None = auto-estimate)
    FLOW_THRESHOLD = 0.4  # Lower = detect more objects (range: 0-1)
    CELLPROB_THRESHOLD = 0.0  # Lower = detect more objects (range: -6 to 6)
    USE_GPU = False  # Set to True if you have CUDA-enabled GPU
    
    # Tracking parameters
    MAX_LINK_DISTANCE = 10  # Max pixel distance to link particles between slices
    MIN_TRACK_LENGTH = 3  # Minimum number of slices a particle must appear in
    
    # ====================================
    
    print("=" * 70)
    print("CELLPOSE-BASED NANOPARTICLE DETECTION")
    print("=" * 70)
    
    image_path = Path(IMAGE_PATH)
    
    # Load image
    print(f"\nLoading: {image_path.name}")
    img = load_zstack(IMAGE_PATH)
    
    # Segment with Cellpose
    masks_3d, estimated_diameter = segment_with_cellpose(
        img,
        model_type=MODEL_TYPE,
        diameter=DIAMETER,
        channels=None,  # Grayscale z-stack
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        use_gpu=USE_GPU
    )
    
    # Get particles per slice
    print("\nExtracting centroids...")
    centroids_per_slice, particles_per_slice = get_particles_per_slice(masks_3d)
    print(f"\nTotal detections across all slices: {sum(particles_per_slice)}")
    
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
        print(f"Detection method: Cellpose ({MODEL_TYPE})")
        print(f"Estimated diameter: {estimated_diameter:.1f} pixels")
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
    mask_path = image_path.parent / f"{image_path.stem}_cellpose_masks.tif"
    tifffile.imwrite(mask_path, masks_3d.astype(np.uint16))
    print(f"\nSaved masks to: {mask_path}")
    
    # Visualizations
    viz_path = image_path.parent / f"{image_path.stem}_cellpose_viz.png"
    visualize_results(img, masks_3d, centroids_per_slice, tracks, output_path=viz_path)
    
    count_path = image_path.parent / f"{image_path.stem}_cellpose_counts.png"
    visualize_particle_count_changes(centroids_per_slice, output_path=count_path)
    
    return tracks, properties


if __name__ == "__main__":
    main()
