# Nanoparticle Counting with Cellpose

This project uses cellpose to count unique nanoparticles in z-stack images, handling particles that go in and out of focus.

## Setup

Your cellpose environment is located at: `~/cellpose-env`

To activate:
```bash
source ~/cellpose-env/bin/activate
```

## Quick Start - Counting Particles

### 1. Count particles in your z-stack:

```bash
cd ~/nanoparticle-tracking
source ~/cellpose-env/bin/activate
python count_nanoparticles.py
```

**Current Result:** 16 unique particles detected in your test file

### 2. Adjust parameters if needed

Edit `count_nanoparticles.py` and modify these lines (~205-214):

```python
# Segmentation parameters
DIAMETER = 15  # Increase if particles are larger, decrease if smaller
FLOW_THRESHOLD = 0.4  # Higher = stricter (fewer merged particles)
CELLPROB_THRESHOLD = 0.0  # Lower = detect fainter objects

# Linking parameters
MAX_LINK_DISTANCE = 20  # Max pixel distance to link between slices
MIN_TRACK_LENGTH = 3  # Minimum slices a particle must appear in
```

**Parameter tuning tips:**
- If particles are being missed: decrease `CELLPROB_THRESHOLD` to -1 or -2
- If detecting too many false positives: increase `FLOW_THRESHOLD` to 0.5-0.6
- If particles aren't linking properly: increase `MAX_LINK_DISTANCE` to 30-40
- If too many noise detections: increase `MIN_TRACK_LENGTH` to 4-5

### 3. Process different files

Change `IMAGE_PATH` in the script (line 203):
```python
IMAGE_PATH = "/path/to/your/zstack.tif"
```

## Training a Custom Model

The pretrained `cyto2` model may not be optimal for nanoparticles. You can train a custom model!

### Step 1: Create training data

**Option A: Using Cellpose GUI (Recommended)**
```bash
source ~/cellpose-env/bin/activate
cellpose --gui
```

1. Load a z-stack
2. Go through slices and manually segment particles (or use model then correct)
3. Save masks (File → Save masks)
4. Copy annotated files to `~/nanoparticle-tracking/training_data/`

**Option B: Using existing segmentations**
If you already have manual segmentations from ImageJ/Fiji:
1. Export as 16-bit TIFF
2. Each particle should have unique label (1, 2, 3, ...)
3. Background = 0
4. Save with naming: `image.tif` and `image_masks.tif`

### Step 2: Train the model

```bash
cd ~/nanoparticle-tracking
source ~/cellpose-env/bin/activate

# First run creates training_data directory and shows guide
python train_custom_model.py

# After adding training data, run again to train
python train_custom_model.py
```

**Training recommendations:**
- Start with 10-20 manually annotated slices
- Include variety: different focus levels, densities, sizes
- Train for 200-500 epochs (takes 10-30 minutes)
- Test on new data
- Add more training data if needed and retrain

### Step 3: Use your custom model

After training completes, update `count_nanoparticles.py`:
```python
MODEL_TYPE = '/path/to/your/trained_models/nanoparticle_model'
```

## Output Files

Each run generates:
1. **`*_tracked_masks.tif`** - 3D mask where each unique particle has the same ID across all slices
2. **`*_tracking_viz.png`** - Visualization showing detected particles at different z-levels

## Workflow for Multiple Samples

### Process many files at once:

Create a batch script `batch_count.py`:

```python
from pathlib import Path
import subprocess

# Directory containing your z-stacks
data_dir = Path("~/Documents/UROP_Peng").expanduser()

# Find all TIFF files
tiff_files = list(data_dir.rglob("*.tif"))

results = {}
for tif_file in tiff_files:
    print(f"\nProcessing: {tif_file.name}")
    
    # Update IMAGE_PATH in count_nanoparticles.py
    # Then run it
    # Parse output to get particle count
    # Store in results dictionary
    
# Export results to CSV
import pandas as pd
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('particle_counts.csv')
```

## Understanding the Algorithm

### How it works:

1. **2D Segmentation**: Each z-slice is segmented independently
   - Better for anisotropic z-stacks (where z-spacing >> particle size)
   - Detects particles at their best focus

2. **3D Linking**: Detected particles are linked across slices
   - Uses Hungarian algorithm for optimal matching
   - Links particles based on proximity (MAX_LINK_DISTANCE)
   - Handles particles appearing/disappearing

3. **Track Filtering**: Short tracks are removed
   - Filters out noise and spurious detections
   - Only counts particles appearing in ≥ MIN_TRACK_LENGTH slices

4. **Unique Counting**: Each track = one unique particle
   - Solves the multiple-counting problem
   - Particles counted once even if they span many slices

## Troubleshooting

### Too many particles detected
- Increase `FLOW_THRESHOLD` (0.5-0.6)
- Increase `MIN_TRACK_LENGTH` (4-6)
- Increase `CELLPROB_THRESHOLD` (0.5-1.0)

### Missing particles
- Decrease `CELLPROB_THRESHOLD` (-2 to -1)
- Decrease `FLOW_THRESHOLD` (0.2-0.3)
- Increase `MAX_LINK_DISTANCE` (30-50)
- Check `DIAMETER` parameter matches your particles

### Particles not linking correctly
- Increase `MAX_LINK_DISTANCE`
- Check for drift in your z-stack (may need drift correction first)

### Segmentation quality poor
- Train a custom model with your data!
- Pretrained models aren't optimized for nanoparticles

## Scripts Overview

- **`count_nanoparticles.py`** - Main counting script (configured for your data)
- **`count_particles_2d_linking.py`** - Generic template for other datasets
- **`train_custom_model.py`** - Training workflow for custom models

## Current Test Results

File: `2025107_R8_1pBSA_1pCasein_cell5_163_cropped.tif`
- Z-stack: 100 slices, 256×256 pixels
- Total detections: 111 (across all slices)
- Unique particles: **16**
- Average track length: 5.8 ± 2.6 slices
- Longest track: 13 slices (particle ID 18, slices 49-61)

Outputs saved to:
- `2025107_R8_1pBSA_1pCasein_cell5_163_cropped_tracked_masks.tif`
- `2025107_R8_1pBSA_1pCasein_cell5_163_cropped_tracking_viz.png`

## Next Steps

1. **Validate results**: Check the visualization PNG to verify counting accuracy
2. **Optimize parameters**: Adjust if over/under-counting
3. **Train custom model**: Use 10-20 manual annotations for better accuracy
4. **Scale up**: Process all your z-stacks with optimized parameters
5. **Export results**: Collect counts into spreadsheet for analysis

## Questions?

- Check visualization to validate counting
- Adjust parameters based on your specific particles
- Training a custom model will give the best results for your specific nanoparticles
# nanoparticle_quantification
