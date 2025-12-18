#!/usr/bin/env python3
"""Quick diagnostic to identify detection issues"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load test TIFF and extract frame 30
    test_file = Path("/Users/cindyli/nanoparticle-tracking/training_data_cropped/2025107_R8_1to500to100_cell4_208_cropped.tif")
    
    print(f"Testing: {test_file.name}, frame 30\n")
    
    # Load TIFF stack
    import tifffile
    stack = tifffile.imread(test_file)
    img = stack[30]  # Get frame 30
    
    print("=" * 60)
    print("IMAGE PROPERTIES")
    print("=" * 60)
    print(f"Shape: {img.shape}")
    print(f"Dtype: {img.dtype}")
    print(f"Range: [{img.min()}, {img.max()}]")
    print(f"Mean: {img.mean():.1f}, Median: {np.median(img):.1f}")
    print(f"Std: {img.std():.1f}")
    
    # Test blob detection (SAM2's approach)
    print("\n" + "=" * 60)
    print("BLOB DETECTION TEST (SAM2 approach)")
    print("=" * 60)
    
    from skimage.feature import blob_log
    
    # Normalize
    img_norm = (img - img.min()) / (img.max() - img.min())
    
    # Use upscaled image for blob detection (matching SAM2 script)
    h, w = img.shape
    upscaled = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    upscaled_norm = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min())
    
    # Test with higher threshold for better precision
    sam2_threshold = 0.05
    sam2_blobs = blob_log(
        upscaled_norm,
        min_sigma=2,
        max_sigma=20,
        num_sigma=10,
        threshold=sam2_threshold,
        overlap=0.5
    )
    print(f"Blob detection (threshold={sam2_threshold}): {len(sam2_blobs)} blobs detected")
    
    # Test Cellpose quickly
    print("\n" + "=" * 60)
    print("CELLPOSE TEST")
    print("=" * 60)
    
    cellpose_masks = None
    try:
        from cellpose import models
        
        print(f"Upscaled: {img.shape} -> {upscaled.shape}")
        
        model = models.CellposeModel(gpu=False, model_type='cyto2')
        
        # Use automatic diameter estimation (diameter=None)
        print(f"\nTesting with automatic diameter estimation...")
        
        cellpose_masks, _, _ = model.eval(
            upscaled,
            diameter=None,  # Auto-estimate
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        
        n_detected = len(np.unique(cellpose_masks)) - 1
        print(f"  -> Detected: {n_detected} particles")
        
        if n_detected == 0:
            print("  ⚠️  ZERO DETECTIONS!")
                
    except Exception as e:
        print(f"Cellpose error: {e}")
    
    # Test MEDIAR
    print("\n" + "=" * 60)
    print("MEDIAR TEST")
    print("=" * 60)
    
    mediar_masks = None
    try:
        import torch
        import torch.nn as nn
        from monai.inferers import sliding_window_inference
        from segmentation_models_pytorch import MAnet
        from scipy.ndimage import label
        
        # MEDIAR Model Definition
        class MEDIARFormer(MAnet):
            def __init__(self, encoder_name="mit_b5", encoder_weights="imagenet", 
                         decoder_channels=(1024, 512, 256, 128, 64), decoder_pab_channels=256, 
                         in_channels=3, classes=3):
                super().__init__(encoder_name=encoder_name, encoder_weights=encoder_weights, 
                                 decoder_channels=decoder_channels, decoder_pab_channels=decoder_pab_channels, 
                                 in_channels=in_channels, classes=classes)
                self.segmentation_head = None
                self._convert_activations(self.encoder, nn.ReLU, nn.Mish(inplace=True))
                self._convert_activations(self.decoder, nn.ReLU, nn.Mish(inplace=True))
                self.cellprob_head = self._create_head(decoder_channels[-1], 1)
                self.gradflow_head = self._create_head(decoder_channels[-1], 2)
            
            def _create_head(self, in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, in_c // 2, 3, padding=1), nn.Mish(inplace=True),
                    nn.BatchNorm2d(in_c // 2), nn.Conv2d(in_c // 2, out_c, 3, padding=1)
                )
            
            def _convert_activations(self, module, from_ac, to_ac):
                for name, child in module.named_children():
                    if isinstance(child, from_ac): setattr(module, name, to_ac)
                    else: self._convert_activations(child, from_ac, to_ac)
            
            def forward(self, x):
                features = self.encoder(x)
                decoder_output = self.decoder(*features)
                masks = torch.cat([self.gradflow_head(decoder_output), self.cellprob_head(decoder_output)], dim=1)
                return masks
        
        # Load MEDIAR weights
        weights_path = "/Users/cindyli/nanoparticle-tracking/mediar_pretrained_weights/from_phase2.pth"
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Loading MEDIAR model on {device}...")
        model = MEDIARFormer()
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        model.to(device).eval()
        
        # Preprocess upscaled image
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        upscaled_rgb = upscaled_rgb.astype(np.float32) / 65535.0 if upscaled.dtype == np.uint16 else upscaled_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(upscaled_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = sliding_window_inference(tensor, (512, 512), 4, model, overlap=0.6, mode="gaussian")
        
        # Post-process
        pred = output.cpu().numpy()[0]
        cellprob = 1.0 / (1.0 + np.exp(-pred[2]))
        mask_scaled, _ = label(cellprob > 0.5)
        
        # Downscale
        mediar_masks = cv2.resize(mask_scaled.astype(np.float32), (img.shape[1], img.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        
        n_detected = len(np.unique(mediar_masks)) - 1
        print(f"  -> Detected: {n_detected} particles")
        
    except Exception as e:
        print(f"MEDIAR not available or error: {e}")
    
    # Test SAM2 (optional - if available)
    print("\n" + "=" * 60)
    print("SAM2 TEST")
    print("=" * 60)
    
    sam2_masks = None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        
        checkpoint = "/Users/cindyli/nanoparticle-tracking/checkpoints/sam2_hiera_small.pt"
        config = "sam2_hiera_s.yaml"
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading SAM2 model on {device}...")
        model = build_sam2(config, checkpoint, device=device)
        model.to(dtype=torch.float32)
        predictor = SAM2ImagePredictor(model)
        
        # Convert to RGB uint8
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)
        mi, ma = upscaled_rgb.min(), upscaled_rgb.max()
        if ma > mi:
            upscaled_rgb = (255 * (upscaled_rgb - mi) / (ma - mi)).astype(np.uint8)
        else:
            upscaled_rgb = upscaled_rgb.astype(np.uint8)
        
        predictor.set_image(upscaled_rgb)
        
        # Use detected blobs as prompts
        sam2_masks_composite = np.zeros(upscaled_rgb.shape[:2], dtype=np.uint16)
        label_id = 1
        
        print(f"Running SAM2 on {len(sam2_blobs)} prompts...")
        for i, blob in enumerate(sam2_blobs[:30]):  # Limit to 30 for speed
            y, x, r = blob
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                if scores[0] > 0.5:
                    mask_bool = masks[0].astype(bool)
                    sam2_masks_composite[mask_bool] = label_id
                    label_id += 1
            except Exception as e:
                continue
        
        n_detected = label_id - 1
        print(f"  -> Detected: {n_detected} particles")
        sam2_masks = sam2_masks_composite
        
    except Exception as e:
        print(f"SAM2 not available or error: {e}")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Blob detection (SAM2 approach)
    axes[0, 1].imshow(img, cmap='gray')
    if len(sam2_blobs) > 0:
        # Downscale blob coordinates back to original size
        for blob in sam2_blobs:
            y, x, r = blob
            y_orig, x_orig = y / 4, x / 4
            r_orig = r / 4
            c = plt.Circle((x_orig, y_orig), r_orig*np.sqrt(2), color='red', 
                         linewidth=1.5, fill=False, alpha=0.7)
            axes[0, 1].add_patch(c)
    axes[0, 1].set_title(f'Blob Detection (SAM2 approach)\n{len(sam2_blobs)} blobs', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Cellpose
    axes[1, 0].imshow(img, cmap='gray', alpha=0.7)
    if cellpose_masks is not None:
        # Downscale masks back to original size
        masks_down = cv2.resize(cellpose_masks.astype(np.float32), 
                               (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        axes[1, 0].imshow(masks_down, cmap='tab20', alpha=0.5, vmin=0, vmax=20)
        n_cp = len(np.unique(cellpose_masks)) - 1
        axes[1, 0].set_title(f'Cellpose (diam=auto)\n{n_cp} particles', 
                            fontsize=12, fontweight='bold')
    else:
        axes[1, 0].set_title('Cellpose\nNot available', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # MEDIAR
    axes[1, 1].imshow(img, cmap='gray', alpha=0.7)
    if mediar_masks is not None:
        axes[1, 1].imshow(mediar_masks, cmap='tab20', alpha=0.5, vmin=0, vmax=20)
        n_mediar = len(np.unique(mediar_masks)) - 1
        axes[1, 1].set_title(f'MEDIAR (upscaled 4x)\n{n_mediar} particles', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].set_title('MEDIAR\nNot available', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # SAM2
    axes[1, 2].imshow(img, cmap='gray', alpha=0.7)
    if sam2_masks is not None:
        # Downscale masks back to original size
        sam2_masks_down = cv2.resize(sam2_masks.astype(np.float32), 
                                     (img.shape[1], img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        axes[1, 2].imshow(sam2_masks_down, cmap='tab20', alpha=0.5, vmin=0, vmax=20)
        n_sam2 = len(np.unique(sam2_masks)) - 1
        axes[1, 2].set_title(f'SAM2\n{n_sam2} particles', fontsize=12, fontweight='bold')
    else:
        axes[1, 2].set_title('SAM2\nNot available', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_dir = Path("/Users/cindyli/nanoparticle-tracking/debug_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "detection_comparison_208_frame30.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
