# -*- coding: utf-8 -*-
"""Data processing functions for Landsat data."""

import copy
import numpy as np
import torch
from typing import Dict, Tuple

from config import (
    LANDSAT_5_MAX_IDX, LANDSAT_8_9_MAX_IDX,
    LANDSAT_5_THERMAL_BAND_IDX, LANDSAT_8_9_THERMAL_BAND_IDX
)


def is_landsat_eight(curr_tensor: torch.Tensor) -> bool:
    """Check if tensor is from Landsat 8/9 (11 bands).
    
    Args:
        curr_tensor: Tensor with shape (stations, bands, height, width)
        
    Returns:
        True if Landsat 8/9, False otherwise
    """
    return curr_tensor.shape[1] == 11


def is_landsat_five(curr_tensor: torch.Tensor) -> bool:
    """Check if tensor is from Landsat 5 (7 bands).
    
    Args:
        curr_tensor: Tensor with shape (stations, bands, height, width)
        
    Returns:
        True if Landsat 5, False otherwise
    """
    return curr_tensor.shape[1] == 7


def coefficients_from_metadata(metadata: Dict) -> Tuple[Dict, Dict]:
    """Extract radiometric and thermal constants from metadata.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Tuple of (radiometric_rescaling, thermal_constants)
    """
    return (
        metadata['LANDSAT_METADATA_FILE']["LEVEL1_RADIOMETRIC_RESCALING"],
        metadata['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS']
    )


def convert_to_brightness_temperature(data: Dict) -> Dict:
    """Convert Landsat data to brightness temperature.
    
    This function converts digital numbers to radiance and then to brightness temperature
    for thermal bands. The conversion formulas are:
    - Radiance: Lλ = ML * Qcal + AL
    - Landsat 5 BT: BT = K2 / ln(K1 / Lλ + 1)
    - Landsat 8/9 BT: BT = K2 / (K1 / Lλ + 1)
    
    Args:
        data: Dictionary mapping scene names to their data
        
    Returns:
        Dictionary with converted brightness temperature tensors
    """
    data_processed = copy.deepcopy(data)
    scenes_to_remove = []
    
    for scene_date, data_ in data_processed.items():
        curr_tensor = data_["tensor"][:, :, :, :]
        
        # Determine Landsat type
        landsat_max_index = LANDSAT_5_MAX_IDX + 1
        if is_landsat_eight(curr_tensor):
            landsat_max_index = LANDSAT_8_9_MAX_IDX + 1
        elif not is_landsat_five(curr_tensor):
            print(f"Invalid Landsat image tensor shape - scene {scene_date}, shape: {curr_tensor.shape}")
            scenes_to_remove.append(scene_date)
            continue
        
        try:
            coeffs, level_one_coeffs = coefficients_from_metadata(data_["metadata"])
        except KeyError as e:
            print(f"Missing metadata keys for scene {scene_date}: {e}")
            scenes_to_remove.append(scene_date)
            continue
        
        # Convert to double precision for calculations
        curr_tensor = curr_tensor.type(torch.DoubleTensor)
        
        # Convert all bands to radiance
        for i in range(1, landsat_max_index):
            curr_tensor[:, i-1, :, :] = (
                curr_tensor[:, i-1, :, :] * float(coeffs[f"RADIANCE_MULT_BAND_{i}"]) +
                float(coeffs[f"RADIANCE_ADD_BAND_{i}"])
            )
        
        # Convert thermal band to brightness temperature
        if is_landsat_eight(curr_tensor):
            # Landsat 8/9: BT = K2 / (K1 / Lλ + 1)
            curr_tensor[:, LANDSAT_8_9_THERMAL_BAND_IDX, :, :] = (
                float(level_one_coeffs['K2_CONSTANT_BAND_10']) /
                (float(level_one_coeffs['K1_CONSTANT_BAND_10']) / 
                 (curr_tensor[:, 9, :, :] + 1))
            )
        elif is_landsat_five(curr_tensor):
            # Landsat 5: BT = K2 / ln(K1 / Lλ + 1)
            curr_tensor[:, LANDSAT_5_THERMAL_BAND_IDX, :, :] = (
                float(level_one_coeffs['K2_CONSTANT_BAND_6']) /
                np.log(float(level_one_coeffs['K1_CONSTANT_BAND_6']) / 
                       curr_tensor[:, 5, :, :] + 1)
            )
        else:
            print(f"Something went wrong with scene {scene_date}")
            scenes_to_remove.append(scene_date)
            continue
        
        data_processed[scene_date]['tensor'] = curr_tensor
    
    # Remove scenes that failed processing
    for key in scenes_to_remove:
        data_processed.pop(key, None)
    
    return data_processed

