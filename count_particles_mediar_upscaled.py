"""
MEDIAR-based Nanoparticle Segmentation with Image Upscaling

Key Features:
- Input: 256x256 with ~15-pixel diameter particles
- Upscale by 4x by default (1024x1024)
- Process with MEDIAR
- Downscale results back to original resolution
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import tifffile
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import zoom
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check if MEDIAR is installed
try:
    from monai.inferers import sliding_window_inference
    from segmentation_models_pytorch import MAnet
except ImportError as e:
    print("ERROR: Required MEDIAR dependencies not installed.")
    print("\nPlease activate the virtual environment:")
    print("  source activate_mediar.sh")
    sys.exit(1)


class MEDIARFormer(MAnet):
    """MEDIAR-Former Model"""

    def __init__(
        self,
        encoder_name="mit_b5",
        encoder_weights="imagenet",
        decoder_channels=(1024, 512, 256, 128, 64),
        decoder_pab_channels=256,
        in_channels=3,
        classes=3,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_pab_channels=decoder_pab_channels,
            in_channels=in_channels,
            classes=classes,
        )
        self.segmentation_head = None
        self._convert_activations(self.encoder, nn.ReLU, nn.Mish(inplace=True))
        self._convert_activations(self.decoder, nn.ReLU, nn.Mish(inplace=True))
        self.cellprob_head = self._create_segmentation_head(
            in_channels=decoder_channels[-1], out_channels=1
        )
        self.gradflow_head = self._create_segmentation_head(
            in_channels=decoder_channels[-1], out_channels=2
        )

    def _create_segmentation_head(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def _convert_activations(self, module, from_activation, to_activation):
        for name, child in module.named_children():
            if isinstance(child, from_activation):
                setattr(module, name, to_activation)
            else:
                self._convert_activations(child, from_activation, to_activation)

    def forward(self, x):
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        cellprob_mask = self.cellprob_head(decoder_output)
        gradflow_mask = self.gradflow_head(decoder_output)
        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)
        return masks


class MEDIARPredictorUpscaled:
    """MEDIAR Predictor with automatic upscaling for small particles"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_tta: bool = False,
        upscale_factor: float = 4.0,
        roi_size: int = 512,
        overlap: float = 0.6,
    ):
        """
        Initialize MEDIAR predictor with upscaling
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on
            use_tta: Whether to use test-time augmentation
            upscale_factor: Factor to upscale images (e.g., 4.0 = 4x larger)
            roi_size: Size of sliding window patches
            overlap: Overlap ratio between patches
        """
        self.device = torch.device(device)
        self.use_tta = use_tta
        self.upscale_factor = upscale_factor
        self.roi_size = roi_size
        self.overlap = overlap

        print(f"Initializing MEDIARFormer with {upscale_factor}x upscaling...")
        self.model = MEDIARFormer()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
        print(f"Loading weights from: {model_path}")
        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using bicubic interpolation"""
        if self.upscale_factor == 1.0:
            return image
        
        h, w = image.shape[:2]
        new_h = int(h * self.upscale_factor)
        new_w = int(w * self.upscale_factor)
        
        if len(image.shape) == 2:
            upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return upscaled

    def downscale_mask(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Downscale segmentation mask back to original size using nearest neighbor"""
        if self.upscale_factor == 1.0:
            return mask
        
        # Use nearest neighbor to preserve label IDs
        downscaled = cv2.resize(
            mask.astype(np.float32), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        return downscaled.astype(mask.dtype)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for MEDIAR inference"""
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0

        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference with upscaling
        
        Args:
            image: Input image (original resolution)
            
        Returns:
            Tuple of (instance_mask, flow_field, cell_probability) at original resolution
        """
        original_shape = image.shape[:2]
        
        # Upscale input
        upscaled_image = self.upscale_image(image)
        print(f"  Upscaled from {original_shape} to {upscaled_image.shape[:2]}")
        
        # Preprocess
        image_tensor = self.preprocess_image(upscaled_image).to(self.device)
        
        with torch.no_grad():
            # Sliding window inference
            pred_mask = sliding_window_inference(
                inputs=image_tensor,
                roi_size=self.roi_size,
                sw_batch_size=4,
                predictor=self.model,
                overlap=self.overlap,
                mode="gaussian",
            )

            # Apply TTA if enabled
            if self.use_tta:
                pred_hflip = sliding_window_inference(
                    inputs=image_tensor.flip(3),
                    roi_size=self.roi_size,
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=self.overlap,
                    mode="gaussian",
                ).flip(3)

                pred_vflip = sliding_window_inference(
                    inputs=image_tensor.flip(2),
                    roi_size=self.roi_size,
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=self.overlap,
                    mode="gaussian",
                ).flip(2)

                pred_mask_merged = torch.zeros_like(pred_mask)
                pred_mask_merged[:, 0] = (pred_mask[:, 0] + pred_hflip[:, 0] - pred_vflip[:, 0]) / 3
                pred_mask_merged[:, 1] = (pred_mask[:, 1] - pred_hflip[:, 1] + pred_vflip[:, 1]) / 3
                pred_mask_merged[:, 2] = (pred_mask[:, 2] + pred_hflip[:, 2] + pred_vflip[:, 2]) / 3
                pred_mask = pred_mask_merged

        # Extract outputs
        pred_mask = pred_mask.cpu().numpy()[0]
        
        flow_y = pred_mask[0]
        flow_x = pred_mask[1]
        cellprob = 1.0 / (1.0 + np.exp(-pred_mask[2]))
        
        # Compute instance masks at upscaled resolution
        instance_mask_upscaled = self._compute_masks(
            flow_field=(flow_y, flow_x),
            cellprob=cellprob,
            flow_threshold=0.4,
            cellprob_threshold=0.5,
        )
        
        # Downscale back to original resolution
        instance_mask = self.downscale_mask(instance_mask_upscaled, original_shape)
        cellprob_downscaled = cv2.resize(cellprob, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        flow_y_downscaled = cv2.resize(flow_y, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        flow_x_downscaled = cv2.resize(flow_x, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        flow_field = np.stack([flow_y_downscaled, flow_x_downscaled], axis=-1)
        
        return instance_mask, flow_field, cellprob_downscaled

    def _compute_masks(
        self,
        flow_field: Tuple[np.ndarray, np.ndarray],
        cellprob: np.ndarray,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.5,
    ) -> np.ndarray:
        """Compute instance masks from flow field and cell probability"""
        from scipy.ndimage import label
        mask = cellprob > cellprob_threshold
        instance_mask, num_instances = label(mask)
        return instance_mask


# =====================
# Z-STACK UTILITIES
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


def get_track_properties(tracks: Dict[int, List[Tuple[int, float, float]]]):
    props = {}
    for tid, pos in tracks.items():
        zs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        xs = [p[2] for p in pos]
        props[tid] = {
            'n_slices': len(pos),
            'z_range': (int(min(zs)), int(max(zs))),
            'z_extent': int(max(zs) - min(zs) + 1),
            'centroid': (float(np.mean(zs)), float(np.mean(ys)), float(np.mean(xs))),
            'trajectory': pos,
        }
    return props


def mediar_segment_zstack(img_zyx: np.ndarray, predictor: MEDIARPredictorUpscaled) -> np.ndarray:
    """Segment each z-slice using MEDIAR predictor"""
    Z = img_zyx.shape[0]
    masks_3d = np.zeros(img_zyx.shape, dtype=np.uint16)
    for z in tqdm(range(Z), desc="Segmenting z-slices with MEDIAR (upscaled)"):
        slice2d = img_zyx[z]
        inst_mask, _, _ = predictor.predict(slice2d)
        masks_3d[z] = inst_mask.astype(np.uint16)
    return masks_3d


def visualize_mediar_zstack(img_zyx: np.ndarray, masks_3d: np.ndarray, centroids_per_slice, tracks, save_path: str, sample_interval: int = 5):
    """Visualize results, sampling every N slices"""
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
        axes[0, i].imshow(img_zyx[z], cmap='gray')
        axes[0, i].set_title(f'Z={z}')
        axes[0, i].axis('off')
        axes[1, i].imshow(img_zyx[z], cmap='gray', alpha=0.6)
        axes[1, i].imshow(masks_3d[z], cmap='tab20', alpha=0.4, vmin=0, vmax=20)
        axes[1, i].set_title(f'Detected: {len(centroids_per_slice[z])}')
        axes[1, i].axis('off')
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
    parser = argparse.ArgumentParser(
        description="MEDIAR with upscaling for nanoparticle segmentation"
    )
    parser.add_argument('--image', '-i', type=str, 
                        default='training_data_cropped',
                        help='Path to TIFF file or directory')
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='Path to MEDIAR weights (.pth)')
    parser.add_argument('--output', '-o', type=str, 
                        default='training_data_output',
                        help='Output directory')
    parser.add_argument('--device', '-d', type=str, default='cpu', 
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for inference')
    parser.add_argument('--upscale-factor', '-u', type=float, default=5.0,
                        help='Image upscaling factor (e.g., 4.0 = 4x larger, makes 15px → 60px)')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--max-link-distance', type=float, default=10.0,
                        help='Max distance to link particles between slices')
    parser.add_argument('--min-track-length', type=int, default=3,
                        help='Minimum track length')
    parser.add_argument('--viz-sample-interval', type=int, default=5,
                        help='Sample every Nth slice for visualization (default: 5)')

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

    # Initialize predictor with upscaling
    predictor = MEDIARPredictorUpscaled(
        model_path=args.weights,
        device=args.device,
        use_tta=args.tta,
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
            # Z-stack processing
            masks_3d = mediar_segment_zstack(stack, predictor)
            centroids_per_slice, counts = get_particles_per_slice(masks_3d)
            tracks = link_particles_across_slices(centroids_per_slice, max_distance=args.max_link_distance)
            tracks = filter_short_tracks(tracks, min_length=args.min_track_length)

            # Save results
            mask_path = out_dir / f"{Path(p).stem}_mediar_upscaled_masks.tif"
            tifffile.imwrite(str(mask_path), masks_3d.astype(np.uint16))
            
            if not args.no_viz:
                viz_path = out_dir / f"{Path(p).stem}_mediar_upscaled_viz.png"
                visualize_mediar_zstack(stack, masks_3d, centroids_per_slice, tracks, 
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
