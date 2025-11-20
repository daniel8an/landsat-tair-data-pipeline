# -*- coding: utf-8 -*-
"""Data loading functions for Landsat data processing."""

import os
import json
import torch
import pandas as pd
from typing import Dict, List, Tuple

from config import (
    BASE_DIR, TENSORS_DIR, METADATA_DIR, GROUND_TRUTHS_FILE
)


def read_station_file(station_path: str) -> List[int]:
    """Read station IDs from a text file.
    
    Args:
        station_path: Path to the station file
        
    Returns:
        List of station IDs
    """
    with open(station_path, "r") as f:
        station_txt = f.read()
    
    res = [int(x[:-1]) if "," in x else int(x) for x in station_txt[1:-1].split(" ")]
    return res


def read_metadata(metadata_path: str) -> Dict:
    """Read metadata from a JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        
    Returns:
        Dictionary containing metadata
    """
    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)
    return metadata_dict


def read_ground_truths(landsat_scene: str, stations: List[int], df_gt: pd.DataFrame) -> List[float]:
    """Read ground truth air temperatures for given stations and date.
    
    Args:
        landsat_scene: Landsat scene identifier (e.g., "LC08_L1TP_174038_20190603_20200828_02_T1")
        stations: List of station IDs
        df_gt: DataFrame containing ground truth data
        
    Returns:
        List of air temperatures corresponding to each station
    """
    datetime_ = landsat_scene.split("_")[3]
    year = datetime_[:4]
    month = datetime_[4:6]
    day = datetime_[6:]
    
    gt_arr = []
    for station_id in stations:
        winner = df_gt[
            (df_gt.year == int(year)) & 
            (df_gt.month == int(month)) & 
            (df_gt.day == int(day)) & 
            (df_gt.station_id == int(station_id))
        ]
        if len(winner) > 0:
            gt_arr.append(winner.air_temp.iloc[0])
        else:
            gt_arr.append(-9999.0)  # Missing value indicator
    
    return gt_arr


def load_ground_truths_dataframe() -> pd.DataFrame:
    """Load and preprocess the ground truths CSV file.
    
    Returns:
        DataFrame with year, month, and day columns added
    """
    df_gt = pd.read_csv(os.path.join(BASE_DIR, GROUND_TRUTHS_FILE))
    
    # Extract year, month and day
    dates = pd.to_datetime(df_gt.utc_date)
    df_gt['month'] = dates.apply(lambda x: x.month)
    df_gt['year'] = dates.apply(lambda x: x.year)
    df_gt['day'] = dates.apply(lambda x: x.day)
    
    return df_gt


def get_tensor_names() -> Tuple[List[str], List[str]]:
    """Get lists of tensor and station file names.
    
    Returns:
        Tuple of (tensor_names, station_names)
    """
    tensors_dir_path = os.path.join(BASE_DIR, TENSORS_DIR)
    tensors_dir_content = os.listdir(tensors_dir_path)
    
    tensor_names = [x for x in tensors_dir_content if ".pt" in x]
    station_names = [x for x in tensors_dir_content if ".txt" in x]
    
    return tensor_names, station_names


def load_landsat_data(tensor_names: List[str], df_gt: pd.DataFrame, 
                      skip_first: bool = True) -> Dict:
    """Load all Landsat data into a dictionary.
    
    Args:
        tensor_names: List of tensor file names
        df_gt: DataFrame containing ground truth data
        skip_first: Whether to skip the first tensor (default: True)
        
    Returns:
        Dictionary mapping scene names to their data (tensor, stations, metadata, ground_truths)
    """
    data_dict = {}
    tensors_dir_path = os.path.join(BASE_DIR, TENSORS_DIR)
    metadata_dir_path = os.path.join(BASE_DIR, METADATA_DIR)
    
    tensor_list = tensor_names[1:] if skip_first else tensor_names
    
    for idx, tensor_name in enumerate(tensor_list):
        tensor_path = os.path.join(tensors_dir_path, tensor_name)
        
        try:
            tensor_ = torch.load(tensor_path)
            tensor_ = tensor_.permute(1, 0, 2, 3)
        except Exception as e:
            print(f"Could not load tensor {tensor_name}: {e}")
            continue
        
        landsat_scene = tensor_name.split(".")[0]
        station_name = f"{landsat_scene}_stations.txt"
        station_path = os.path.join(tensors_dir_path, station_name)
        
        try:
            stations = read_station_file(station_path)
        except Exception as e:
            print(f"Could not load stations {landsat_scene}: {e}")
            continue
        
        metadata_path = os.path.join(metadata_dir_path, f"{landsat_scene}_MTL_metadata.json")
        
        try:
            metadata = read_metadata(metadata_path)
        except Exception as e:
            print(f"Could not load metadata {landsat_scene}: {e}")
            continue
        
        try:
            ground_truths = read_ground_truths(landsat_scene, stations, df_gt)
        except Exception as e:
            print(f"Could not load ground truths {landsat_scene}: {e}")
            continue
        
        data_dict[landsat_scene] = {
            "tensor": tensor_,
            "stations": stations,
            "metadata": metadata,
            "ground_truths": ground_truths
        }
    
    return data_dict

