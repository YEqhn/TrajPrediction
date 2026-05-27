import torch
import numpy as np
import os
import pandas as pd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def load_dataset_data(base_path, train_ratio=0.8, window_seconds=1.0, predict_ms=100, sample_rate=200):
    """
    Load data from data_processed CSV files (ab06_ramp folder structure)
    
    Args:
        base_path: path to data_processed folder
        train_ratio: ratio of training data
        window_seconds: input window duration in seconds (default: 1.0)
        predict_ms: prediction horizon in milliseconds (default: 100)
        sample_rate: sampling frequency in Hz (default: 200)
    
    Returns:
        train_dataset, test_dataset, input_size, output_size, normalization_params
    """
    ramp_path = os.path.join(base_path, 'ab06_ramp_csv')
    imu_path = os.path.join(ramp_path, 'imu')
    gon_path = os.path.join(ramp_path, 'gon')
    
    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"IMU path does not exist: {imu_path}")
    if not os.path.exists(gon_path):
        raise FileNotFoundError(f"GON path does not exist: {gon_path}")
    
    window_samples = int(window_seconds * sample_rate)
    predict_samples = int(predict_ms * sample_rate / 1000)
    
    print("=" * 60)
    print("Dataset Loading Configuration")
    print("=" * 60)
    print(f"Data path: {ramp_path}")
    print(f"Input window: {window_seconds}s ({window_samples} samples at {sample_rate}Hz)")
    print(f"Prediction horizon: {predict_ms}ms ({predict_samples} samples)")
    print("=" * 60)
    
    imu_files = sorted([f for f in os.listdir(imu_path) if f.endswith('.csv')])
    gon_files = sorted([f for f in os.listdir(gon_path) if f.endswith('.csv')])
    
    matched_files = sorted(list(set(imu_files) & set(gon_files)))
    
    print(f"Found {len(matched_files)} matched file pairs")
    
    if not matched_files:
        raise ValueError("No matching IMU/GON files found")
    
    all_sequences = []
    imu_feature_names = None
    gon_feature_names = None
    
    for data_file in matched_files:
        imu_file_path = os.path.join(imu_path, data_file)
        gon_file_path = os.path.join(gon_path, data_file)
        
        try:
            imu_df = pd.read_csv(imu_file_path)
            gon_df = pd.read_csv(gon_file_path)
            
            if imu_feature_names is None:
                imu_feature_names = list(imu_df.columns[1:])
                gon_feature_names = list(gon_df.columns[1:])
            
            imu_data = imu_df.iloc[:, 1:].values
            
            sagittal_indices = []
            for col_name in gon_feature_names:
                if 'sagittal' in col_name.lower():
                    sagittal_indices.append(gon_feature_names.index(col_name))
            
            if len(sagittal_indices) < 3:
                raise ValueError(f"GON data must contain at least 3 sagittal angles. Found: {[gon_feature_names[i] for i in sagittal_indices]}")
            
            target_columns = [gon_feature_names[i] for i in sagittal_indices]
            gon_data = gon_df[target_columns].values
            
            min_len = min(len(imu_data), len(gon_data))
            imu_data = imu_data[:min_len]
            gon_data = gon_data[:min_len]
            
            file_sequences = create_sliding_window_sequences(
                imu_data, gon_data, 
                window_samples, predict_samples
            )
            
            if file_sequences:
                all_sequences.extend(file_sequences)
                print(f"[LOADED] {data_file}: {len(file_sequences)} sequences")
        
        except Exception as e:
            print(f"[ERROR] Failed to load {data_file}: {e}")
            continue
    
    if not all_sequences:
        raise ValueError("No valid data found")
    
    input_size = all_sequences[0][0].shape[0]
    output_size = all_sequences[0][1].shape[0]
    
    input_data = np.array([seq[0].numpy() for seq in all_sequences])
    output_data = np.array([seq[1].numpy() for seq in all_sequences])
    
    print(f"\nTotal sequences: {len(all_sequences)}")
    print(f"Input shape: {input_data.shape} (sequences x features x time)")
    print(f"Output shape: {output_data.shape} (sequences x joint angles)")
    print(f"Input features ({len(imu_feature_names)}): {imu_feature_names}")
    print(f"Output targets (sagittal only, {len(target_columns)}): {target_columns}")
    
    input_mean = np.mean(input_data, axis=(0, 2), keepdims=True)
    input_std = np.std(input_data, axis=(0, 2), keepdims=True)
    output_mean = np.mean(output_data, axis=0, keepdims=True)
    output_std = np.std(output_data, axis=0, keepdims=True)
    
    input_std[input_std < 1e-8] = 1.0
    output_std[output_std < 1e-8] = 1.0
    
    input_data_norm = (input_data - input_mean) / input_std
    output_data_norm = (output_data - output_mean) / output_std
    
    n_samples = input_data_norm.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_ratio)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_input = torch.tensor(input_data_norm[train_indices], dtype=torch.float32)
    train_output = torch.tensor(output_data_norm[train_indices], dtype=torch.float32)
    test_input = torch.tensor(input_data_norm[test_indices], dtype=torch.float32)
    test_output = torch.tensor(output_data_norm[test_indices], dtype=torch.float32)
    
    print(f"\nData split: {train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% test")
    print(f"Train samples: {train_input.shape[0]}, Test samples: {test_input.shape[0]}")
    
    train_dataset = list(zip(train_input, train_output))
    test_dataset = list(zip(test_input, test_output))
    
    norm_params = {
        'input_mean': input_mean.squeeze(),
        'input_std': input_std.squeeze(),
        'output_mean': output_mean.squeeze(),
        'output_std': output_std.squeeze(),
        'imu_features': imu_feature_names,
        'gon_features': gon_feature_names,
        'window_samples': window_samples,
        'predict_samples': predict_samples,
        'sample_rate': sample_rate
    }
    
    return train_dataset, test_dataset, input_size, output_size, norm_params


def create_sliding_window_sequences(imu_data, gon_data, window_samples, predict_samples):
    """
    Create sliding window sequences for training
    
    Args:
        imu_data: IMU sensor data (N x features)
        gon_data: Goniometer joint angle data (N x joints)
        window_samples: number of samples in input window
        predict_samples: number of samples ahead to predict
    
    Returns:
        list of (input_tensor, output_tensor) tuples
    """
    sequences = []
    n_samples = len(imu_data)
    
    if n_samples < window_samples + predict_samples:
        return []
    
    for i in range(n_samples - window_samples - predict_samples + 1):
        seq_input = imu_data[i:i+window_samples]
        seq_output = gon_data[i+window_samples+predict_samples-1]
        
        seq_input_tensor = torch.tensor(seq_input.transpose(), dtype=torch.float32)
        seq_output_tensor = torch.tensor(seq_output, dtype=torch.float32)
        
        sequences.append((seq_input_tensor, seq_output_tensor))
    
    return sequences


def data_generator(dataset, batch_size, shuffle=True):
    """
    Create data loader from dataset
    """
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = [dataset[j] for j in batch_indices]
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        yield inputs, targets


def denormalize_predictions(predictions, norm_params):
    """
    Denormalize model predictions back to original scale (degrees)
    
    Args:
        predictions: normalized predictions tensor (batch_size x output_size)
        norm_params: dictionary containing output_mean and output_std
    
    Returns:
        denormalized predictions in original units (degrees)
    """
    output_mean = torch.tensor(norm_params['output_mean'], dtype=torch.float32)
    output_std = torch.tensor(norm_params['output_std'], dtype=torch.float32)
    
    denormalized = predictions * output_std + output_mean
    
    return denormalized


def denormalize_error(mse_value, output_std):
    """
    Denormalize MSE error to original scale (degrees²)
    
    Args:
        mse_value: MSE in normalized space
        output_std: standard deviation used for normalization
    
    Returns:
        denormalized MSE in original units
    """
    return mse_value * (output_std ** 2)


def denormalize_mae(mae_value, output_std):
    """
    Denormalize MAE error to original scale (degrees)
    
    Args:
        mae_value: MAE in normalized space
        output_std: standard deviation used for normalization
    
    Returns:
        denormalized MAE in original units
    """
    return mae_value * output_std


def denormalize_rmse(rmse_value, output_std):
    """
    Denormalize RMSE error to original scale (degrees)
    
    Args:
        rmse_value: RMSE in normalized space
        output_std: standard deviation used for normalization
    
    Returns:
        denormalized RMSE in original units
    """
    return rmse_value * output_std


def calculate_rmse_from_mse(mse_value):
    """
    Calculate RMSE from MSE value
    
    Args:
        mse_value: MSE value
    
    Returns:
        RMSE value
    """
    return np.sqrt(mse_value)