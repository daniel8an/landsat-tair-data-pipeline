# Landsat Data Processing Pipeline

A comprehensive Python pipeline for processing Landsat satellite imagery data and extracting features for machine learning applications, specifically for air temperature prediction using ground truth measurements from IMS (Israel Meteorological Service) stations.

## Overview

This project processes Landsat 5, 8, and 9 satellite imagery to extract features that can be used for predicting air temperature. The pipeline includes:

- **Data Loading**: Loading Landsat tensor data, metadata, and ground truth measurements
- **Data Processing**: Converting digital numbers to radiance and brightness temperature
- **Feature Extraction**: Extracting spatial and radiometric features from satellite imagery
- **Data Augmentation**: Applying various augmentation techniques to increase dataset size
- **Train/Test Splitting**: Preparing data for machine learning model training

## Project Structure

```
landsat_processing/
├── config.py                 # Configuration file with paths and constants
├── data_loader.py            # Functions for loading Landsat data and ground truths
├── data_processor.py         # Functions for converting to brightness temperature
├── feature_extractor.py      # Functions for extracting ML features
├── data_augmentation.py      # Data augmentation functions
├── main.py                   # Main pipeline script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── ground_truths_combined.csv  # Ground truth air temperature data
├── landsat_tensors_and_ground_truths/  # Directory containing .pt tensor files and .txt station files
└── metadatas/                # Directory containing JSON metadata files
```

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The required packages are:
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `torch>=1.9.0`

## Usage

### Basic Usage

Run the main pipeline script:

```bash
python main.py
```

This will execute the complete pipeline:
1. Load ground truth data from CSV
2. Discover and list available tensor files
3. Load Landsat data (tensors, metadata, stations)
4. Convert to brightness temperature
5. Extract features for machine learning
6. Apply data augmentation
7. Split into train/test sets

### Configuration

Edit `config.py` to customize:
- **BASE_DIR**: Base directory for data files (default: ".")
- **TENSORS_DIR**: Directory containing tensor files
- **METADATA_DIR**: Directory containing metadata JSON files
- **GROUND_TRUTHS_FILE**: Path to ground truth CSV file
- **TRAIN_TEST_SPLIT**: Ratio for train/test split (default: 0.8)

### Data Augmentation

The pipeline includes several data augmentation techniques:
- **Horizontal Flip**: Flips spatial features horizontally
- **Vertical Flip**: Flips spatial features vertically
- **90° Rotation**: Rotates spatial features 90 degrees
- **Noise Addition**: Adds Gaussian noise to features
- **Brightness Adjustment**: Adjusts feature brightness

You can customize augmentations in `main.py` by modifying the `augmentations` parameter in the `apply_augmentations()` call.

## Data Format

### Input Data

- **Tensor Files** (`.pt`): PyTorch tensors containing Landsat image data
  - Shape: `(bands, stations, height, width)` before processing
  - Processed to: `(stations, bands, height, width)`
  
- **Station Files** (`.txt`): Text files containing station IDs for each scene
  - Format: `[station_id1, station_id2, ...]`

- **Metadata Files** (`.json`): JSON files containing Landsat metadata
  - Includes radiometric rescaling coefficients and thermal constants

- **Ground Truth CSV**: CSV file with columns:
  - `utc_date`: Date of measurement
  - `station_id`: IMS station ID
  - `air_temp`: Air temperature in Celsius

### Output Data

The pipeline returns a dictionary containing:
- `X_train`, `X_test`: Feature matrices (numpy arrays)
- `y_train`, `y_test`: Target values (air temperature)
- `scenes_train`, `scenes_test`: Scene identifiers
- `stations_train`, `stations_test`: Station IDs

## Features

The extracted features include:
1. **Spatial Features**: Flattened 7×7 pixel windows from each spectral band
2. **Radiometric Coefficients**: Multiply and add coefficients for each band
3. **Thermal Constants**: K1 and K2 constants for brightness temperature conversion
4. **Landsat Type Indicator**: Binary flag indicating Landsat 5 vs 8/9

## Landsat Support

The pipeline supports:
- **Landsat 5**: 7 spectral bands
- **Landsat 8/9**: 11 spectral bands (mapped to 7 bands for consistency)

## IMS Stations

The project uses data from IMS (Israel Meteorological Service) weather stations located throughout Israel. Station information including coordinates is defined in `config.py`.

## Notes

- The pipeline skips the first tensor file by default (set `skip_first=True` in `load_landsat_data()`)
- Missing ground truth values are marked with `-9999.0` and excluded from feature extraction
- Scenes with invalid tensor shapes or missing metadata are automatically filtered out
- Data augmentation is applied before train/test splitting to increase training data diversity

## Google Colab Usage

If running in Google Colab, uncomment the following lines in `main.py`:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Also update `BASE_DIR` in `config.py` to point to your Colab data directory.

## License

Free Software License

## Author

Daniel Eitan, Asher Holder,  Zohar Yakhini and Alexnadra Chudnovsky

