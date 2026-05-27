import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        from TimesNet_Model.layers.Conv_Blocks import Inception_Block_V1
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = self._FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res

    def _FFT_for_Period(self, x, k=2):
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]


class TimesNetAnomalyDetector(nn.Module):
    def __init__(self, input_channels: int, seq_len: int = 100, d_model: int = 32, 
                 d_ff: int = 128, e_layers: int = 2, top_k: int = 3, num_kernels: int = 6, dropout: float = 0.1):
        super(TimesNetAnomalyDetector, self).__init__()
        
        class Config:
            pass
        self.config = Config()
        self.config.seq_len = seq_len
        self.config.pred_len = 0
        self.config.d_model = d_model
        self.config.d_ff = d_ff
        self.config.e_layers = e_layers
        self.config.top_k = top_k
        self.config.num_kernels = num_kernels
        self.config.dropout = dropout
        
        from TimesNet_Model.layers.Embed import DataEmbedding
        self.enc_embedding = DataEmbedding(input_channels, d_model, 'timeF', 'h', dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.blocks = nn.ModuleList([TimesBlock(self.config) for _ in range(e_layers)])
        self.projection = nn.Linear(d_model, input_channels, bias=True)
        
        self.reconstruction_history = deque(maxlen=100)
        self.baseline_threshold = None
        
    def forward(self, x_enc):
        B, T, N = x_enc.size()
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)
        
        enc_out = self.enc_embedding(x_enc, None)
        
        for i in range(self.config.e_layers):
            enc_out = self.layer_norm(self.blocks[i](enc_out))
        
        dec_out = self.projection(enc_out)
        
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, T, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, T, 1))
        
        return dec_out
    
    def compute_anomaly_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reconstruction = self.forward(x)
        scores = F.mse_loss(reconstruction, x, reduction='none').mean(dim=-1)
        return scores, reconstruction
    
    def update_baseline(self, normal_data: torch.Tensor):
        self.eval()
        with torch.no_grad():
            scores = []
            for i in range(0, len(normal_data), 32):
                batch = normal_data[i:i+32]
                if batch.dim() == 2:
                    batch = batch.unsqueeze(0)
                elif batch.dim() == 3 and batch.size(0) != 1:
                    for j in range(batch.size(0)):
                        sub_batch = batch[j:j+1]
                        score, _ = self.compute_anomaly_score(sub_batch)
                        scores.append(score.item())
                else:
                    score, _ = self.compute_anomaly_score(batch)
                    scores.append(score.item())
            
            all_scores = np.array(scores)
            self.baseline_threshold = float(np.percentile(all_scores, 95))
            self.baseline_mean = float(np.mean(all_scores))
            self.baseline_std = float(np.std(all_scores))
    
    def get_normalized_score(self, score: float) -> float:
        if self.baseline_std > 0:
            return (score - self.baseline_mean) / self.baseline_std
        return score - self.baseline_mean
    
    def is_anomalous(self, score: torch.Tensor, threshold_multiplier: float = 1.5) -> bool:
        if self.baseline_threshold is None:
            return False
        threshold = self.baseline_threshold * threshold_multiplier
        return score.item() > threshold


class IMUDataLoader:
    def __init__(self, data_path: str, seq_len: int = 100, sample_rate: int = 200):
        self.data_path = data_path
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.ramp_path = os.path.join(data_path, 'ab06_ramp_csv')
        self.imu_path = os.path.join(self.ramp_path, 'imu')
        
    def load_imu_files(self) -> List[str]:
        if os.path.exists(self.imu_path):
            return sorted([f for f in os.listdir(self.imu_path) if f.endswith('.csv')])
        return []
    
    def load_imu_data(self, filename: str) -> np.ndarray:
        filepath = os.path.join(self.imu_path, filename)
        df = pd.read_csv(filepath)
        return df.iloc[:, 1:].values
    
    def create_sequences(self, data: np.ndarray) -> List[torch.Tensor]:
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            seq = data[i:i+self.seq_len]
            sequences.append(torch.tensor(seq, dtype=torch.float32))
        return sequences
    
    def prepare_training_data(self) -> Tuple[torch.Tensor, List[str]]:
        files = self.load_imu_files()
        all_sequences = []
        file_labels = []
        
        for filename in files:
            data = self.load_imu_data(filename)
            sequences = self.create_sequences(data)
            all_sequences.extend(sequences)
            file_labels.extend([filename] * len(sequences))
        
        if all_sequences:
            return torch.stack(all_sequences), file_labels
        return torch.tensor([]), []


class AnomalyInjector:
    def __init__(self, noise_level: float = 0.5, dropout_prob: float = 0.1, spike_magnitude: float = 5.0):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.spike_magnitude = spike_magnitude
        
    def inject_sensor_noise(self, data: torch.Tensor, intensity: float = 1.0) -> torch.Tensor:
        noise = torch.randn_like(data) * self.noise_level * intensity
        return data + noise
    
    def inject_data_dropout(self, data: torch.Tensor, drop_ratio: float = 0.2) -> torch.Tensor:
        mask = torch.rand_like(data) > drop_ratio
        return data * mask.float()
    
    def inject_spike_anomaly(self, data: torch.Tensor, spike_ratio: float = 0.1) -> torch.Tensor:
        result = data.clone()
        B, T, N = data.shape
        
        num_spikes = int(T * spike_ratio)
        for b in range(B):
            for n in range(N):
                spike_indices = torch.randperm(T)[:num_spikes]
                result[b, spike_indices, n] += torch.randn(num_spikes) * self.spike_magnitude
        
        return result
    
    def inject_drift(self, data: torch.Tensor, drift_rate: float = 0.01) -> torch.Tensor:
        result = data.clone()
        B, T, N = data.shape
        
        for b in range(B):
            for n in range(N):
                t = torch.arange(T, dtype=torch.float32)
                drift = t * drift_rate * torch.randn(1) * data[b, :, n].std()
                result[b, :, n] += drift
        
        return result
    
    def inject_freeze(self, data: torch.Tensor, freeze_ratio: float = 0.15) -> torch.Tensor:
        result = data.clone()
        B, T, N = data.shape
        
        freeze_start = int(T * 0.4)
        freeze_end = freeze_start + int(T * freeze_ratio)
        
        for b in range(B):
            freeze_value = data[b, freeze_start - 1, :].clone()
            for t in range(freeze_start, freeze_end):
                result[b, t, :] = freeze_value + torch.randn(N) * 0.1
        
        return result
    
    def inject_random_anomaly_type(self, data: torch.Tensor) -> Tuple[torch.Tensor, str]:
        anomaly_type = np.random.choice(['noise', 'dropout', 'spike', 'drift', 'freeze', 'normal'], 
                                       p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
        
        if anomaly_type == 'noise':
            return self.inject_sensor_noise(data, intensity=np.random.uniform(1.0, 3.0)), 'sensor_noise'
        elif anomaly_type == 'dropout':
            return self.inject_data_dropout(data, drop_ratio=np.random.uniform(0.15, 0.35)), 'data_dropout'
        elif anomaly_type == 'spike':
            return self.inject_spike_anomaly(data, spike_ratio=np.random.uniform(0.05, 0.2)), 'spike_anomaly'
        elif anomaly_type == 'drift':
            return self.inject_drift(data, drift_rate=np.random.uniform(0.005, 0.02)), 'sensor_drift'
        elif anomaly_type == 'freeze':
            return self.inject_freeze(data, freeze_ratio=np.random.uniform(0.1, 0.25)), 'data_freeze'
        else:
            return data, 'normal'


def create_timesnet_configs(seq_len: int, input_channels: int) -> object:
    class Config:
        pass
    config = Config()
    config.seq_len = seq_len
    config.pred_len = 0
    config.d_model = 32
    config.d_ff = 128
    config.e_layers = 2
    config.top_k = 3
    config.num_kernels = 6
    config.dropout = 0.1
    config.enc_in = input_channels
    config.dec_in = input_channels
    config.c_out = input_channels
    config.embed = 'timeF'
    config.freq = 'h'
    return config


def train_timesnet_anomaly_detector(
    data_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cuda',
    save_path: str = None
) -> TimesNetAnomalyDetector:
    from torch.utils.data import DataLoader, TensorDataset
    
    print("=" * 80)
    print("TimesNet Anomaly Detector Training")
    print("=" * 80)
    
    imu_loader = IMUDataLoader(data_path)
    data, labels = imu_loader.prepare_training_data()
    
    if len(data) == 0:
        raise ValueError("No IMU data found for training")
    
    input_channels = data.shape[-1]
    seq_len = data.shape[-2]
    
    print(f"Data shape: {data.shape}")
    print(f"Input channels: {input_channels}")
    print(f"Sequence length: {seq_len}")
    print(f"Total samples: {len(data)}")
    
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    configs = create_timesnet_configs(seq_len, input_channels)
    model = TimesNetAnomalyDetector(
        input_channels=input_channels,
        seq_len=seq_len,
        d_model=configs.d_model,
        d_ff=configs.d_ff,
        e_layers=configs.e_layers,
        top_k=configs.top_k,
        num_kernels=configs.num_kernels,
        dropout=configs.dropout
    )
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_data, in train_loader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            reconstruction = model(batch_data)
            loss = criterion(reconstruction, batch_data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    model.eval()
    model.update_baseline(train_data.to(device))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_channels': input_channels,
                'seq_len': seq_len,
                'd_model': configs.d_model,
                'd_ff': configs.d_ff,
                'e_layers': configs.e_layers,
            },
            'baseline_threshold': model.baseline_threshold,
            'baseline_mean': model.baseline_mean,
            'baseline_std': model.baseline_std
        }, save_path)
        print(f"Model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/yeqhn/Desktop/TCN/data_processed')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default='./models/timesnet_anomaly_detector.pt')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = train_timesnet_anomaly_detector(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        save_path=args.save_path
    )
    
    print("Training completed!")