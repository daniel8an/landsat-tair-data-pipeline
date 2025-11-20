# -*- coding: utf-8 -*-
"""Data augmentation functions for Landsat data.
Automatically applies rotations (90, 180, 270 degrees) with coordinate and temporal adjustments.
"""

import numpy as np
import random
from typing import Tuple, Optional
from geopy.distance import geodesic


def rotate_tensor(tensor: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a 3D tensor by degrees.
    
    Args:
        tensor: 3D tensor with shape (bands, height, width) or (height, width, bands)
        degrees: Rotation angle (90, 180, or 270)
        
    Returns:
        Rotated tensor
    """
    if degrees == 90:
        return np.rot90(tensor, k=1, axes=(1, 2))
    elif degrees == 180:
        return np.rot90(tensor, k=2, axes=(1, 2))
    elif degrees == 270:
        return np.rot90(tensor, k=3, axes=(1, 2))
    else:
        raise ValueError("Only 90, 180, and 270 degrees rotations are supported")


def adjust_day_month_v2(day: float, month: float) -> Tuple[int, int]:
    """Adjust day and month with random shifts.
    
    Args:
        day: Original day
        month: Original month
        
    Returns:
        Tuple of (adjusted_day, adjusted_month)
    """
    day_shift = random.randint(5, 15)  # Shift day by 5-15 days
    month_shift = 0 if random.random() > 0.7 else 1  # Occasionally shift month
    
    new_day = (int(float(day)) + day_shift) % 30
    if new_day == 0:
        new_day = 1
    
    new_month = (int(float(month)) + month_shift) % 12
    if new_month == 0:
        new_month = 1
    
    return new_day, new_month


def move_right_bottom(longitude: float, latitude: float, distance_right: float, distance_bottom: float) -> Tuple[float, float]:
    """Move coordinates right and bottom by specified distances in meters.
    
    Args:
        longitude: Original longitude
        latitude: Original latitude
        distance_right: Distance to move right (in meters)
        distance_bottom: Distance to move bottom (in meters)
        
    Returns:
        Tuple of (new_longitude, new_latitude)
    """
    # Calculate the distance in meters that corresponds to one degree of longitude at the given latitude
    lon_deg_to_m = geodesic((latitude, longitude), (latitude, longitude + 1)).meters
    
    # Calculate the distance in meters that corresponds to one degree of latitude
    lat_deg_to_m = geodesic((latitude, longitude), (latitude + 1, longitude)).meters
    
    # Calculate the new longitude after moving right by the specified distance
    new_longitude = longitude + (distance_right / lon_deg_to_m)
    
    # Calculate the new latitude after moving bottom by the specified distance
    new_latitude = latitude - (distance_bottom / lat_deg_to_m)
    
    return new_longitude, new_latitude


def move_upper_left(longitude: float, latitude: float, distance_left: float, distance_top: float) -> Tuple[float, float]:
    """Move coordinates left and top by specified distances in meters.
    
    Args:
        longitude: Original longitude
        latitude: Original latitude
        distance_left: Distance to move left (in meters)
        distance_top: Distance to move top (in meters)
        
    Returns:
        Tuple of (new_longitude, new_latitude)
    """
    # Calculate the distance in meters that corresponds to one degree of longitude at the given latitude
    lon_deg_to_m = geodesic((latitude, longitude), (latitude, longitude - 1)).meters
    
    # Calculate the distance in meters that corresponds to one degree of latitude
    lat_deg_to_m = geodesic((latitude, longitude), (latitude + 1, longitude)).meters
    
    # Calculate the new longitude after moving left by the specified distance
    new_longitude = longitude - (distance_left / lon_deg_to_m)
    
    # Calculate the new latitude after moving top by the specified distance
    new_latitude = latitude + (distance_top / lat_deg_to_m)
    
    return new_longitude, new_latitude


def move_randomly(longitude: float, latitude: float, max_shift_km: float) -> Tuple[float, float]:
    """Move coordinates randomly in one of four directions.
    
    Args:
        longitude: Original longitude
        latitude: Original latitude
        max_shift_km: Maximum shift distance in kilometers
        
    Returns:
        Tuple of (new_longitude, new_latitude)
    """
    direction = random.choice(['right_bottom', 'left_top', 'right_top', 'left_bottom'])
    shift_x = random.uniform(5, max_shift_km) * 1000  # Convert to meters
    shift_y = random.uniform(5, max_shift_km) * 1000
    
    if direction == 'right_bottom':
        return move_right_bottom(longitude, latitude, shift_x, shift_y)
    elif direction == 'left_top':
        return move_upper_left(longitude, latitude, shift_x, shift_y)
    elif direction == 'right_top':
        new_long, new_lat = move_right_bottom(longitude, latitude, shift_x, -shift_y)
        return new_long, new_lat
    elif direction == 'left_bottom':
        new_long, new_lat = move_upper_left(longitude, latitude, shift_x, -shift_y)
        return new_long, new_lat


def apply_augmentations(X_train: np.ndarray, y_train: np.ndarray,
                       random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply data augmentation to training data only.
    
    Creates 3 augmented versions (90°, 180°, 270° rotations) plus original = 4x total.
    Each augmentation includes:
    - Image rotation
    - Coordinate adjustment (longitude, latitude)
    - Temporal adjustment (day, month)
    
    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training target values (n_samples,)
        random_seed: Random seed for reproducibility (default: None)
        
    Returns:
        Tuple of (X_train_augmented, y_train_augmented)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Constants
    num_image_features = 7 * 7 * 7  # 7 bands * 7 height * 7 width
    num_total_features = X_train.shape[1]
    num_other_features = num_total_features - num_image_features
    
    new_X_train = []
    new_y_train = []
    
    for train_x_instance, train_y_instance in zip(X_train, y_train):
        # Split features into image and other features
        image_part = train_x_instance[:num_image_features]
        image_tensor = image_part.reshape(7, 7, 7)  # (bands, height, width)
        other_features = train_x_instance[-num_other_features:].copy()
        
        # Extract original coordinates and temporal info
        # Feature order: ..., longitude, latitude, year, month, day
        # So: day=other_features[-1], month=other_features[-2], year=other_features[-3]
        # long=other_features[-5], lat=other_features[-4]
        original_day = other_features[-1]
        original_month = other_features[-2]
        original_year = other_features[-3]
        curr_long, curr_lat = float(other_features[-5]), float(other_features[-4])
        
        # Adjust day/month for each rotation
        day_90, month_90 = adjust_day_month_v2(original_day, original_month)
        day_180, month_180 = adjust_day_month_v2(original_day, original_month)
        day_270, month_270 = adjust_day_month_v2(original_day, original_month)
        
        # Rotate image tensors
        rotated_90 = rotate_tensor(image_tensor, 90)
        rotated_180 = rotate_tensor(image_tensor, 180)
        rotated_270 = rotate_tensor(image_tensor, 270)
        
        # Flatten rotated images
        flattened_90 = rotated_90.reshape(num_image_features)
        flattened_180 = rotated_180.reshape(num_image_features)
        flattened_270 = rotated_270.reshape(num_image_features)
        
        # Apply random coordinate movement for each augmentation
        long_90, lat_90 = move_randomly(curr_long, curr_lat, max_shift_km=10)
        long_180, lat_180 = move_randomly(curr_long, curr_lat, max_shift_km=15)
        long_270, lat_270 = move_randomly(curr_long, curr_lat, max_shift_km=10)
        
        # Create adjusted feature vectors for each rotation
        # Feature order: ..., longitude, latitude, year, month, day
        adjusted_features_90 = other_features.copy()
        adjusted_features_90[-5], adjusted_features_90[-4] = long_90, lat_90
        adjusted_features_90[-1], adjusted_features_90[-2] = day_90, month_90
        
        adjusted_features_180 = other_features.copy()
        adjusted_features_180[-5], adjusted_features_180[-4] = long_180, lat_180
        adjusted_features_180[-1], adjusted_features_180[-2] = day_180, month_180
        
        adjusted_features_270 = other_features.copy()
        adjusted_features_270[-5], adjusted_features_270[-4] = long_270, lat_270
        adjusted_features_270[-1], adjusted_features_270[-2] = day_270, month_270
        
        # Concatenate rotated images with adjusted features
        augmented_90 = np.concatenate((flattened_90, adjusted_features_90))
        augmented_180 = np.concatenate((flattened_180, adjusted_features_180))
        augmented_270 = np.concatenate((flattened_270, adjusted_features_270))
        
        # Add original sample
        new_X_train.append(train_x_instance)
        new_y_train.append(train_y_instance)
        
        # Add augmented samples
        new_X_train.append(augmented_90)
        new_y_train.append(train_y_instance)
        
        new_X_train.append(augmented_180)
        new_y_train.append(train_y_instance)
        
        new_X_train.append(augmented_270)
        new_y_train.append(train_y_instance)
    
    # Convert to numpy arrays
    X_train_augmented = np.array(new_X_train)
    y_train_augmented = np.array(new_y_train)
    
    return X_train_augmented, y_train_augmented
