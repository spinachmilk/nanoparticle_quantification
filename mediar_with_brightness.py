"""
MEDIAR-based Nanoparticle Quantification with Brightness Analysis

This script couples Deep Learning segmentation (MEDIAR) with Photometric Analysis.
It solves the "stacking" problem by using the total integrated brightness of a 
segmented blob to estimate how many nanoparticles are contained within it.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import tifffile
from pathlib import Path
from typing import Tuple, List, Dict
from skimage import measure, morphology
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# --- MEDIAR DEPENDENCIES ---
try:
    from monai.inferers import sliding_window_inference
    from segmentation_models_pytorch import MAnet
except ImportError:
    print("ERROR: Required MEDIAR dependencies not installed.")
    print("pip install monai segmentation-models-pytorch")
    sys.exit(1)

warnings.filterwarnings('ignore')


# ==========================================
# 1. MEDIAR MODEL DEFINITIONS (from your file)
# ==========================================
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

class MEDIARPredictorUpscaled:
    def __init__(self, model_path, device="cpu", upscale_factor=4.0):
        self.device = torch.device(device)
        self.upscale_factor = upscale_factor
        self.model = MEDIARFormer()
        if not os.path.exists(model_path): raise FileNotFoundError(f"Weights not found: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.to(self.device).eval()

    def predict(self, image):
        # ... (Simplified prediction wrapper) ...
        # Upscale
        h, w = image.shape[:2]
        scaled_img = cv2.resize(image, (int(w*self.upscale_factor), int(h*self.upscale_factor)), interpolation=cv2.INTER_CUBIC)
        
        # Preprocess
        if len(scaled_img.shape) == 2: scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)
        tensor = torch.from_numpy(scaled_img.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = sliding_window_inference(tensor, (512,512), 4, self.model, overlap=0.5, mode="gaussian")
        
        # Post-process
        pred = output.cpu().numpy()[0]
        cellprob = 1.0 / (1.0 + np.exp(-pred[2]))
        
        # Labeling (Instance Segmentation)
        from scipy.ndimage import label
        mask_scaled, _ = label(cellprob > 0.5)
        
        # Downscale
        mask_final = cv2.resize(mask_scaled.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        return mask_final


# ==========================================
# 2. BRIGHTNESS ANALYSIS LOGIC
# ==========================================

def calculate_blob_brightness(img_slice, mask_slice, bg_ring_width=3):
    """
    Calculates the 'Integrated Intensity' for every segmented blob.
    
    Instead of 'fitgaussian' (which fails on irregular shapes), we sum the
    pixels inside the mask and subtract the LOCAL background.
    """
    props = measure.regionprops(mask_slice.astype(int), intensity_image=img_slice)
    blob_data = []

    for p in props:
        # 1. Create a local background mask (ring around the particle)
        # This handles uneven illumination better than a global background
        particle_mask = (mask_slice == p.label)
        dilated = morphology.binary_dilation(particle_mask, morphology.disk(bg_ring_width))
        ring_mask = dilated & ~particle_mask & (mask_slice == 0) # Ring outside particle, not on other particles
        
        # 2. Estimate Local Background
        if np.sum(ring_mask) > 5:
            local_bg = np.median(img_slice[ring_mask])
        else:
            # Fallback if ring is empty (e.g. crowded): use global slice median
            local_bg = np.median(img_slice)

        # 3. Calculate Integrated Intensity (Photometry)
        # Sum of (Pixel Value - Background) for all pixels in mask
        # Clip at 0 to avoid negative light
        pixel_values = img_slice[particle_mask]
        net_flux = np.sum(np.maximum(0, pixel_values - local_bg))
        
        blob_data.append({
            'label': p.label,
            'y': p.centroid[0],
            'x': p.centroid[1],
            'area': p.area,
            'mean_intensity': p.mean_intensity,
            'local_bg': local_bg,
            'integrated_intensity': net_flux  # THIS IS YOUR "MASS" PROXY
        })
        
    return blob_data

def calibrate_and_count(df_blobs):
    """
    Automatic Calibration:
    Assumes the most common particle size (median) is a SINGLE nanoparticle.
    Everything else is a multiple of this unit.
    """
    if df_blobs.empty:
        return df_blobs, 1.0

    # Robust calibration: Median of the brightness distribution
    # We assume >50% of your blobs are single particles.
    unit_brightness = df_blobs['integrated_intensity'].median()
    
    # Calculate counts
    # We use max(1, ...) so even dim particles count as at least 1
    df_blobs['estimated_count'] = (df_blobs['integrated_intensity'] / unit_brightness).round().clip(lower=1).astype(int)
    
    return df_blobs, unit_brightness


# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MEDIAR + Brightness Quantification")
    parser.add_argument('--image', '-i', type=str, required=True, help='Input TIFF file')
    parser.add_argument('--weights', '-w', type=str, required=True, help='MEDIAR weights path')
    parser.add_argument('--output', '-o', type=str, default='output_quantification', help='Output folder')
    parser.add_argument('--upscale', '-u', type=float, default=4.0, help='Upscale factor')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps"
    
    # 1. Initialize MEDIAR
    print(f"Loading MEDIAR on {device}...")
    predictor = MEDIARPredictorUpscaled(args.weights, device=device, upscale_factor=args.upscale)
    
    # 2. Load Data
    print(f"Loading {args.image}...")
    stack = tifffile.imread(args.image)
    if stack.ndim == 2: stack = stack[np.newaxis, ...]
    
    all_blob_data = []
    masks_3d = np.zeros_like(stack, dtype=np.uint16)

    # 3. Process Slice-by-Slice
    print("Running Segmentation & Photometry...")
    for z in tqdm(range(stack.shape[0])):
        img_slice = stack[z]
        
        # A. SEGMENTATION (The "Where")
        mask = predictor.predict(img_slice)
        masks_3d[z] = mask
        
        # B. PHOTOMETRY (The "How Much")
        blobs = calculate_blob_brightness(img_slice, mask)
        
        # Add Z-index metadata
        for b in blobs:
            b['z'] = z
        all_blob_data.extend(blobs)

    # 4. Calibration & Counting
    print("Calibrating brightness...")
    df = pd.DataFrame(all_blob_data)
    
    if not df.empty:
        df, unit_val = calibrate_and_count(df)
        
        # 5. Report
        total_blobs = len(df)
        total_particles = df['estimated_count'].sum()
        
        print("\n" + "="*40)
        print(f"ANALYSIS SUMMARY for {Path(args.image).name}")
        print("="*40)
        print(f"Calibration Unit (1 NP): {unit_val:.2f} intensity units")
        print(f"Total Blobs Detected:    {total_blobs}")
        print(f"Total Particles Counted: {total_particles} (corrected for stacking)")
        print(f"Gain from Correction:    +{total_particles - total_blobs} particles")
        print("="*40)
        
        # Save CSV
        csv_path = Path(args.output) / f"{Path(args.image).stem}_quantification.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved data to {csv_path}")
        
        # Save Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['integrated_intensity'], bins=50, alpha=0.7, label='Observed Brightness')
        plt.axvline(unit_val, color='r', linestyle='--', label='1-mer (Unit)')
        plt.axvline(unit_val*2, color='g', linestyle='--', label='2-mer')
        plt.axvline(unit_val*3, color='y', linestyle='--', label='3-mer')
        plt.xlabel('Integrated Intensity')
        plt.ylabel('Count')
        plt.title('Nanoparticle Brightness Distribution')
        plt.legend()
        plt.savefig(Path(args.output) / f"{Path(args.image).stem}_brightness_hist.png")
        
        # Save Masks
        tifffile.imwrite(Path(args.output) / f"{Path(args.image).stem}_masks.tif", masks_3d)

if __name__ == "__main__":
    main()