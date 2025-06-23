import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from typing import Tuple
import scipy.io

class SParameterDataset(Dataset):
    """Dataset for raw S parameter data without angle conversion"""
    
    def __init__(self, data_path: str, window_size: int, indices: np.ndarray = None, 
                 use_complex: bool = True, normalize: bool = True):
        """
        Initialize S Parameter Dataset
        
        Args:
            data_path: Path to the raw data file
            window_size: Size of the sliding window for time series
            indices: Optional indices to use for train/test split
            use_complex: Whether to use complex S parameters (magnitude + phase)
            normalize: Whether to normalize the S parameters
        """
        self.window_size = window_size
        self.use_complex = use_complex
        self.normalize = normalize
        
        # Load and process the raw data
        self.load_and_process_data(data_path)
        
        # Set up indices
        self.indices = indices if indices is not None else np.arange(len(self.s_parameters) - window_size + 1)
        self.length = len(self.indices)
        
        print(f"S Parameter Dataset initialized:")
        print(f"- Total samples: {len(self.s_parameters)}")
        print(f"- Window size: {window_size}")
        print(f"- Available windows: {self.length}")
        print(f"- S parameter shape: {self.s_parameters.shape}")
        if self.normalize:
            print(f"- S parameters normalized")
    
    def load_and_process_data(self, data_path: str):
        """Load raw data and convert to S parameters"""
        
        if data_path.endswith('.csv'):
            # Load CSV data (analog inputs)
            df = pd.read_csv(data_path)
            
            # Check for required columns
            required_cols = ['TimeStamps', 'AI0', 'AI1', 'AI2', 'AI3', 'AI4', 'AI5']
            if all(col in df.columns for col in required_cols):
                self.process_analog_inputs(df)
            else:
                raise ValueError(f"CSV file must contain columns: {required_cols}")
                
        elif data_path.endswith('.mat'):
            # Load MATLAB data
            mat_data = scipy.io.loadmat(data_path)
            self.process_matlab_data(mat_data)
        else:
            raise ValueError("Unsupported file format. Use .csv or .mat files")
    
    def process_analog_inputs(self, df: pd.DataFrame):
        """Convert analog input data to S parameters"""
        # Extract analog inputs (AI0-AI5)
        analog_data = df[['AI0', 'AI1', 'AI2', 'AI3', 'AI4', 'AI5']].values
        
        # Convert analog inputs to complex S parameters
        # This is a simplified conversion - in practice, you'd need proper calibration
        # and impedance matching considerations
        
        # Method 1: Use pairs of analog inputs as real/imaginary parts
        s11_real, s11_imag = analog_data[:, 0], analog_data[:, 1]
        s21_real, s21_imag = analog_data[:, 2], analog_data[:, 3]
        s12_real, s12_imag = analog_data[:, 4], analog_data[:, 5]
        
        # For demonstration, assume s22 = s11* (conjugate - common in reciprocal networks)
        s22_real, s22_imag = s11_real, -s11_imag
        
        if self.use_complex:
            # Store as complex numbers
            s11 = s11_real + 1j * s11_imag
            s12 = s12_real + 1j * s12_imag
            s21 = s21_real + 1j * s21_imag
            s22 = s22_real + 1j * s22_imag
            
            # Stack into S parameter matrix format [S11, S12, S21, S22]
            # For neural networks, we'll flatten to [S11_real, S11_imag, S12_real, S12_imag, S21_real, S21_imag, S22_real, S22_imag]
            self.s_parameters = np.column_stack([
                s11.real, s11.imag, s12.real, s12.imag,
                s21.real, s21.imag, s22.real, s22.imag
            ])
        else:
            # Use magnitude and phase instead
            s11_mag = np.sqrt(s11_real**2 + s11_imag**2)
            s11_phase = np.arctan2(s11_imag, s11_real)
            s12_mag = np.sqrt(s12_real**2 + s12_imag**2)
            s12_phase = np.arctan2(s12_imag, s12_real)
            s21_mag = np.sqrt(s21_real**2 + s21_imag**2)
            s21_phase = np.arctan2(s21_imag, s21_real)
            s22_mag = np.sqrt(s22_real**2 + s22_imag**2)
            s22_phase = np.arctan2(s22_imag, s22_real)
            
            self.s_parameters = np.column_stack([
                s11_mag, s11_phase, s12_mag, s12_phase,
                s21_mag, s21_phase, s22_mag, s22_phase
            ])
        
        # Normalize if requested
        if self.normalize:
            # For complex case, normalize real and imaginary parts separately
            # For magnitude/phase case, normalize magnitudes to [0,1] and phases to [-1,1]
            if self.use_complex:
                # Normalize real and imaginary parts
                self.s_parameters = self.normalize_complex_data(self.s_parameters)
            else:
                # Normalize magnitude and phase separately
                self.s_parameters = self.normalize_mag_phase_data(self.s_parameters)
        
        # Create target values (for demonstration, predict future S parameters)
        # In practice, you might want to predict specific characteristics or detect recalibration needs
        self.create_targets()
    
    def process_matlab_data(self, mat_data):
        """Process MATLAB data files"""
        # This would depend on the specific structure of your MATLAB files
        # For now, implement basic processing
        if 'data' in mat_data:
            data = mat_data['data']
            # Assume data is in format [samples, channels] where channels represent S parameter components
            if data.shape[1] >= 8:
                self.s_parameters = data[:, :8]  # Take first 8 columns as S parameters
            else:
                raise ValueError("MATLAB data must have at least 8 columns for S parameters")
        else:
            raise ValueError("MATLAB file must contain 'data' field")
        
        if self.normalize:
            if self.use_complex:
                self.s_parameters = self.normalize_complex_data(self.s_parameters)
            else:
                self.s_parameters = self.normalize_mag_phase_data(self.s_parameters)
        
        self.create_targets()
    
    def normalize_complex_data(self, data):
        """Normalize complex S parameter data (real/imaginary format)"""
        # Normalize each component to have zero mean and unit variance
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            normalized_data[:, i] = (data[:, i] - np.mean(data[:, i])) / (np.std(data[:, i]) + 1e-8)
        return normalized_data
    
    def normalize_mag_phase_data(self, data):
        """Normalize magnitude/phase S parameter data"""
        normalized_data = np.copy(data)
        # Normalize magnitudes (even indices) to [0, 1]
        for i in range(0, data.shape[1], 2):
            mag_data = data[:, i]
            normalized_data[:, i] = (mag_data - np.min(mag_data)) / (np.max(mag_data) - np.min(mag_data) + 1e-8)
        
        # Normalize phases (odd indices) to [-1, 1]
        for i in range(1, data.shape[1], 2):
            phase_data = data[:, i]
            normalized_data[:, i] = phase_data / np.pi  # Assuming phases are in [-pi, pi]
        
        return normalized_data
    
    def create_targets(self):
        """Create target values for training"""
        # For demonstration, predict the next S parameter values (time series forecasting)
        # In your case, you might want to predict recalibration needs or other characteristics
        
        # Simple approach: predict next time step
        self.targets = self.s_parameters[1:].copy()  # Shift by one time step
        self.s_parameters = self.s_parameters[:-1]   # Remove last sample to match target length
        
        print(f"Created targets for prediction task:")
        print(f"- Feature shape: {self.s_parameters.shape}")
        print(f"- Target shape: {self.targets.shape}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data_idx = self.indices[idx]
        
        # Get window of S parameter data
        window = self.s_parameters[data_idx:data_idx + self.window_size]
        target = self.targets[data_idx + self.window_size - 1]
        time_index = data_idx + self.window_size - 1
        
        return torch.FloatTensor(window), torch.FloatTensor(target), time_index


class SParameterRecalibrationDataset(Dataset):
    """Dataset for S parameter recalibration detection (similar to angle-based labeling)"""
    
    def __init__(self, data_path: str, window_size: int = 50, threshold: float = 0.1):
        """
        Initialize S Parameter Recalibration Dataset
        
        Args:
            data_path: Path to the raw data file
            window_size: Size of the sliding window
            threshold: Threshold for detecting significant S parameter changes
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Load S parameter data
        base_dataset = SParameterDataset(data_path, 1, use_complex=True, normalize=True)
        self.s_parameters = base_dataset.s_parameters
        
        # Create labeled sequences for recalibration detection
        self.create_labeled_sequences()
    
    def create_labeled_sequences(self):
        """Create sequences with recalibration labels"""
        self.features = []
        self.targets = []
        self.used_indices = []
        
        for i in range(len(self.s_parameters)):
            for a in range(1, min(self.window_size, len(self.s_parameters) - i)):
                # Extract window of S parameters
                feature_window = self.s_parameters[i:i+a+1]
                
                # Pad sequence to fixed length
                feature_window_fixed = self.pad_sequence(feature_window)
                
                # Check if recalibration is needed based on S parameter change
                if self.recalibration_needed(self.s_parameters[i], self.s_parameters[i+a]):
                    self.features.append(feature_window_fixed.flatten())
                    self.targets.append(1)  # Recalibration needed
                    self.used_indices.append((i, i+a))
                    break
                else:
                    self.features.append(feature_window_fixed.flatten())
                    self.targets.append(0)  # No recalibration needed
                    self.used_indices.append((i, i+a))
        
        self.features = np.array(self.features)
        self.targets = np.array(self.targets)
        
        print(f"Created recalibration dataset:")
        print(f"- Total samples: {len(self.features)}")
        print(f"- Recalibration needed: {np.sum(self.targets)}")
        print(f"- No recalibration: {len(self.targets) - np.sum(self.targets)}")
    
    def pad_sequence(self, seq, fixed_length=None):
        """Pad sequence to fixed length"""
        if fixed_length is None:
            fixed_length = self.window_size
        
        padding_needed = fixed_length - len(seq)
        if padding_needed > 0:
            pad = np.zeros((padding_needed, seq.shape[1]))
            padded_seq = np.vstack((pad, seq))
        else:
            padded_seq = seq[-fixed_length:]
        return padded_seq
    
    def recalibration_needed(self, s_param1, s_param2):
        """
        Determine if recalibration is needed based on S parameter change
        
        This uses the magnitude of change in S parameters as a proxy for
        the angular difference used in the original system
        """
        # Calculate the magnitude of change in S parameters
        diff = np.linalg.norm(s_param1 - s_param2)
        return diff >= self.threshold
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.targets[idx]])
    
    def save_labeled_data(self, output_path: str):
        """Save labeled data for later use"""
        output_df = pd.DataFrame({
            "feature": [str(f.tolist()) for f in self.features],
            "target": self.targets
        })
        output_df.to_csv(output_path, index=False)
        print(f"Saved labeled data to {output_path}") 