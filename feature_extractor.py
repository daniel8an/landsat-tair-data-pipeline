# -*- coding: utf-8 -*-
"""Feature extraction functions for Landsat data."""

import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict

from config import (
    LANDSAT_8_INDEXES_TO_FREQUENCY_MAP,
    IMS_STATIONS
)
from data_processor import coefficients_from_metadata


def extract_features(data_dict: Dict, stations_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from Landsat data for machine learning.
    
    This function extracts features from each station observation:
    - Flattened tensor values (7x7 window)
    - Radiometric coefficients (multiply and add for each band)
    - Thermal constants (K1 and K2)
    - Landsat type indicator (5 vs 8/9)
    
    Args:
        data_dict: Dictionary mapping scene names to their data
        stations_df: DataFrame containing station information
        
    Returns:
        Tuple of (X, y, scenes, stations) where:
        - X: Feature matrix (n_samples, n_features)
        - y: Target values (air temperature)
        - scenes: Scene identifiers for each sample
        - stations: Station IDs for each sample
    """
    X = []
    y = []
    scenes = []
    stations = []
    
    for scene_date, data_ in data_dict.items():
        for idx, (gt, station) in enumerate(zip(data_["ground_truths"], data_['stations'])):
            # Skip missing ground truth values
            if gt == -9999.0:
                continue
            
            curr_tensor = data_["tensor"][idx, :, :, :]
            
            # Check if thermal constants are available
            if 'LEVEL1_THERMAL_CONSTANTS' not in data_["metadata"]['LANDSAT_METADATA_FILE']:
                continue
            
            try:
                coeffs, level_one_coeffs = coefficients_from_metadata(data_["metadata"])
            except KeyError:
                continue
            
            # Determine Landsat type and extract coefficients
            is_landsat_5 = 1
            if curr_tensor.shape[0] == 11:
                # Landsat 8/9: select specific bands
                curr_tensor = np.take(curr_tensor, LANDSAT_8_INDEXES_TO_FREQUENCY_MAP, axis=0)
                coeffs_vals = (
                    [coeffs[f'RADIANCE_MULT_BAND_{i}'] 
                     for i in [x+1 for x in LANDSAT_8_INDEXES_TO_FREQUENCY_MAP]] +
                    [coeffs[f'RADIANCE_ADD_BAND_{i}'] 
                     for i in [x+1 for x in LANDSAT_8_INDEXES_TO_FREQUENCY_MAP]]
                )
                is_landsat_5 = 0
            elif curr_tensor.shape[0] == 7:
                # Landsat 5: all 7 bands
                coeffs_vals = (
                    [coeffs[f'RADIANCE_MULT_BAND_{i}'] for i in range(1, 8)] +
                    [coeffs[f'RADIANCE_ADD_BAND_{i}'] for i in range(1, 8)]
                )
            else:
                print(f"Wrong # of bands, found #{curr_tensor.shape[0]} for scene {scene_date}")
                continue
            
            # Extract thermal constants (K1 and K2)
            k_coeffs = [0, 0]
            if 'K2_CONSTANT_BAND_10' in level_one_coeffs:
                k_coeffs = [
                    float(level_one_coeffs['K2_CONSTANT_BAND_10']),
                    float(level_one_coeffs['K1_CONSTANT_BAND_10'])
                ]
            elif 'K2_CONSTANT_BAND_6' in level_one_coeffs:
                k_coeffs = [
                    float(level_one_coeffs['K2_CONSTANT_BAND_6']),
                    float(level_one_coeffs['K1_CONSTANT_BAND_6'])
                ]
            else:
                print(f"Missing thermal constants for scene {scene_date}")
                continue
            
            # Get station coordinates
            station_info = stations_df[stations_df.id == station]
            if len(station_info) == 0:
                continue
            
            longitude = station_info['longitude'].iloc[0]
            latitude = station_info['latitude'].iloc[0]
            
            # Extract date information from scene
            datetime_ = scene_date.split("_")[3]
            year = int(datetime_[:4])
            month = int(datetime_[4:6])
            day = int(datetime_[6:])
            
            # Flatten tensor and concatenate features
            tensor_np = torch.flatten(curr_tensor).numpy()
            curr_array = np.concatenate((tensor_np, coeffs_vals))
            curr_array = np.concatenate((curr_array, k_coeffs))
            curr_array = np.concatenate((curr_array, [is_landsat_5]))
            # Add coordinates and temporal info for augmentation
            # Order: ..., longitude, latitude, ..., month, day
            curr_array = np.concatenate((curr_array, [longitude, latitude, year, month, day]))
            
            X.append(curr_array)
            y.append(gt)
            scenes.append(scene_date)
            stations.append(station)
    
    return np.array(X), np.array(y), np.array(scenes), np.array(stations)


def split_train_test(X: np.ndarray, y: np.ndarray, scenes: np.ndarray, 
                     stations: np.ndarray, train_ratio: float = 0.8,
                     shuffle: bool = True, random_seed: int = None) -> Tuple:
    """Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target values
        scenes: Scene identifiers
        stations: Station IDs
        train_ratio: Ratio of data to use for training (default: 0.8)
        shuffle: Whether to shuffle data before splitting (default: True)
        random_seed: Random seed for reproducibility (default: None)
        
    Returns:
        Tuple of (X_train, y_train, scenes_train, stations_train,
                  X_test, y_test, scenes_test, stations_test)
    """
    if shuffle:
        import random
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        indices = list(range(len(X)))
        random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        scenes = scenes[indices]
        stations = stations[indices]
    
    train_size = int(len(X) * train_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    scenes_train = scenes[:train_size]
    stations_train = stations[:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    scenes_test = scenes[train_size:]
    stations_test = stations[train_size:]
    
    return (X_train, y_train, scenes_train, stations_train,
            X_test, y_test, scenes_test, stations_test)

