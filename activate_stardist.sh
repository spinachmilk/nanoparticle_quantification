#!/bin/bash
# Activate the StarDist virtual environment
source stardist_venv/bin/activate
echo "StarDist environment activated!"
echo "Python version: $(python --version)"
echo "StarDist version: $(python -c 'import stardist; print(stardist.__version__)')"
