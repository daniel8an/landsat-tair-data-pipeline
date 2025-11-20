# -*- coding: utf-8 -*-
"""Main script for Landsat data processing and feature extraction.
This script replicates the logic from Landsat_data_loading.ipynb
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Import modules (using absolute imports for script execution)
from config import BASE_DIR, TRAIN_TEST_SPLIT, IMS_STATIONS
from data_loader import (
    load_ground_truths_dataframe,
    get_tensor_names,
    load_landsat_data
)
from data_processor import convert_to_brightness_temperature
from feature_extractor import extract_features, split_train_test
from data_augmentation import apply_augmentations


def main(use_augmentation: bool = True):
    """Main function to process Landsat data and prepare training/test sets.
    
    Args:
        use_augmentation: Whether to apply data augmentation to training data (default: True)
    """
    
    print("=" * 60)
    print("Landsat Data Processing Pipeline")
    print("=" * 60)
    if use_augmentation:
        print("Data augmentation: ENABLED")
    else:
        print("Data augmentation: DISABLED")
    
    # Step 1: Load ground truth data
    print("\n[1/7] Loading ground truth data...")
    df_gt = load_ground_truths_dataframe()
    print(f"   Loaded {len(df_gt)} ground truth records")
    
    # Step 2: Get tensor names
    print("\n[2/7] Discovering tensor files...")
    tensor_names, station_names = get_tensor_names()
    print(f"   Found {len(tensor_names)} tensor files")
    print(f"   Found {len(station_names)} station files")
    
    # Print date range
    if tensor_names:
        dates = [int(x.split("_")[3]) for x in tensor_names]
        print(f"   Landsat images date range: {min(dates)} - {max(dates)}")
    
    # Step 3: Load Landsat data
    print("\n[3/7] Loading Landsat data...")
    # Note: skip_first=True by default (matches notebook behavior)
    data_dict = load_landsat_data(tensor_names, df_gt, skip_first=True)
    print(f"   Successfully loaded {len(data_dict)} scenes")
    
    # Step 4: Convert to brightness temperature (optional)
    print("\n[4/7] Converting to brightness temperature...")
    data_processed = convert_to_brightness_temperature(data_dict)
    print(f"   Processed {len(data_processed)} scenes")
    
    # Step 5: Extract features
    print("\n[5/7] Extracting features...")
    stations_df = pd.DataFrame(IMS_STATIONS)
    X, y, scenes, stations = extract_features(data_processed, stations_df)
    print(f"   Extracted {len(X)} samples")
    print(f"   Feature vector length: {X.shape[1]}")
    print(f"   Target range: {y.min():.2f} - {y.max():.2f} °C")
    
    # Step 6: Split into train/test sets (before augmentation)
    print("\n[6/7] Splitting into train/test sets...")
    (X_train, y_train, scenes_train, stations_train,
     X_test, y_test, scenes_test, stations_test) = split_train_test(
        X, y, scenes, stations, train_ratio=TRAIN_TEST_SPLIT, shuffle=True
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Step 7: Apply data augmentation to training data only (if enabled)
    if use_augmentation:
        print("\n[7/7] Applying data augmentation to training data...")
        print("   Applying rotations (90°, 180°, 270°) with coordinate and temporal adjustments...")
        X_train_aug, y_train_aug = apply_augmentations(
            X_train, y_train,
            random_seed=42
        )
        print(f"   Original training samples: {len(X_train)}")
        print(f"   Augmented training samples: {len(X_train_aug)}")
        print(f"   Augmentation factor: {len(X_train_aug) / len(X_train):.2f}x")
    else:
        print("\n[7/7] Skipping data augmentation...")
        X_train_aug = X_train
        y_train_aug = y_train
        print(f"   Training samples: {len(X_train_aug)} (no augmentation applied)")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total original samples: {len(X)}")
    print(f"Training samples (after augmentation): {len(X_train_aug)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training/Test ratio: {len(X_train_aug)/len(X_test):.2f}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Unique scenes: {len(np.unique(scenes))}")
    print(f"Unique stations: {len(np.unique(stations))}")
    print(f"Temperature range: {y.min():.2f} - {y.max():.2f} °C")
    print(f"Mean temperature: {y.mean():.2f} °C")
    print(f"Std temperature: {y.std():.2f} °C")
    
    # Save stations dataframe if it doesn't exist
    stations_csv_path = os.path.join(BASE_DIR, "stations_df.csv")
    if not os.path.exists(stations_csv_path):
        stations_df.to_csv(stations_csv_path, index=False)
        print(f"\nSaved stations dataframe to: {stations_csv_path}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return {
        'X_train': X_train_aug,
        'y_train': y_train_aug,
        'X_test': X_test,
        'y_test': y_test,
        'scenes_train': scenes_train,
        'scenes_test': scenes_test,
        'stations_train': stations_train,
        'stations_test': stations_test
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Landsat Data Processing Pipeline for Tair Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python main.py                    # Run with augmentation (default)
                python main.py --no-augmentation  # Run without augmentation
                python main.py -n                 # Short form: run without augmentation
        """
    )
    parser.add_argument(
        '--no-augmentation',
        '-n',
        action='store_true',
        help='Disable data augmentation (default: augmentation is enabled)'
    )
    
    args = parser.parse_args()
    use_augmentation = not args.no_augmentation
    
    results = main(use_augmentation=use_augmentation)

