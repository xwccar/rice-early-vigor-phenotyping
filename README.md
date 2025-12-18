# rice-early-vigor-phenotyping
Early growth vigor phenotyping using UAV LiDAR and multispectral data
# Early Growth Vigor Phenotyping in Rice

This repository provides trained model weights and testing instructions for the study:

"Phenology-oriented evaluation of early growth vigor in rice using time-series UAV LiDAR and multispectral phenotyping"

## Contents
- `model_weights/`: Trained model weights for tiller number estimation, leaf age estimation, and early growth vigor proxy estimation.
- `example_data/`: Example input data for testing model inference.
- `inference/`: Simple inference script and dependency list.

## Requirements
- Python >= 3.8
- PyTorch >= 1.12
- NumPy

## Testing the model
1. Download or clone this repository.
2. Install dependencies:
   ```bash
   pip install -r inference/requirements.txt
