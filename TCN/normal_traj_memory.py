import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict


class TrajectoryMemoryBuffer:
    def __init__(self, max_len: int, trajectory_dim: int):
        self.max_len = max_len
        self.trajectory_dim = trajectory_dim
        self.buffer = deque(maxlen=max_len)
    
    def append(self, trajectory: torch.Tensor):
        if trajectory.dim() == 1:
            trajectory = trajectory.unsqueeze(0)
        self.buffer.append(trajectory.detach().cpu().numpy())
    
    def get_recent(self, n: int) -> torch.Tensor:
        if len(self.buffer) == 0:
            return torch.zeros(n, self.trajectory_dim)
        recent = list(self.buffer)[-n:]
        while len(recent) < n:
            recent.insert(0, recent[0] if recent else np.zeros(self.trajectory_dim))
        return torch.tensor(np.array(recent), dtype=torch.float32)
    
    def get_all(self) -> torch.Tensor:
        if len(self.buffer) == 0:
            return torch.zeros(1, self.trajectory_dim)
        return torch.tensor(np.array(list(self.buffer)), dtype=torch.float32)
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class LSTMCorrector(nn.Module):
    def __init__(self, trajectory_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super(LSTMCorrector, self).__init__()
        self.trajectory_dim = trajectory_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=trajectory_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_size + trajectory_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, trajectory_dim)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, history_trajectories: torch.Tensor, current_prediction: torch.Tensor) -> torch.Tensor:
        if history_trajectories.dim() == 4:
            history_trajectories = history_trajectories.squeeze(0)
        
        if history_trajectories.dim() == 2:
            history_trajectories = history_trajectories.unsqueeze(0)
        
        lstm_out, _ = self.lstm(history_trajectories)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        weighted = lstm_out * attention_weights
        context = weighted.sum(dim=1)
        
        if current_prediction.dim() == 1:
            current_prediction = current_prediction.unsqueeze(0)
        
        if context.dim() == 2 and context.size(0) != current_prediction.size(0):
            context = context.mean(dim=0, keepdim=True)
        
        combined = torch.cat([context, current_prediction], dim=-1)
        
        correction = self.correction_head(combined)
        
        corrected = current_prediction + self.alpha * correction
        
        return corrected


class AnomalyGatingUnit(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super(AnomalyGatingUnit, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, anomaly_score: torch.Tensor) -> torch.Tensor:
        return self.gate_network(anomaly_score)


class NormalTrajMemory:
    def __init__(
        self,
        trajectory_dim: int,
        memory_size: int = 100,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        anomaly_threshold: float = 0.5,
        correction_weight: float = 0.7,
        device: Optional[str] = None
    ):
        self.trajectory_dim = trajectory_dim
        self.memory_size = memory_size
        self.anomaly_threshold = anomaly_threshold
        self.correction_weight = correction_weight
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.memory_buffer = TrajectoryMemoryBuffer(memory_size, trajectory_dim)
        
        self.lstm_corrector = LSTMCorrector(
            trajectory_dim=trajectory_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout
        ).to(self.device)
        
        self.anomaly_gate = AnomalyGatingUnit(hidden_size=32).to(self.device)
        
        self.anomaly_history = deque(maxlen=50)
        
        self.is_initialized = False
        
        self._init_corrector_weights()
    
    def _init_corrector_weights(self):
        for name, param in self.lstm_corrector.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for name, param in self.anomaly_gate.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def set_anomaly_threshold(self, threshold: float):
        self.anomaly_threshold = threshold
    
    def set_correction_weight(self, weight: float):
        self.correction_weight = weight
    
    def update_memory(self, trajectory: torch.Tensor):
        if trajectory.dim() == 0:
            trajectory = trajectory.unsqueeze(0)
        
        self.memory_buffer.append(trajectory.cpu())
        
        if len(self.memory_buffer) >= 10:
            self.is_initialized = True
    
    def get_memory_statistics(self) -> Dict[str, float]:
        if len(self.memory_buffer) < 2:
            return {'mean': 0.0, 'std': 0.0, 'count': len(self.memory_buffer)}
        
        all_trajs = self.memory_buffer.get_all().to(self.device)
        mean = all_trajs.mean(dim=0)
        std = all_trajs.std(dim=0)
        
        return {
            'mean': mean.mean().item(),
            'std': std.mean().item(),
            'count': len(self.memory_buffer),
            'initialized': self.is_initialized
        }
    
    def reset(self):
        self.memory_buffer.clear()
        self.anomaly_history.clear()
        self.is_initialized = False
    
    def correct_trajectory(
        self,
        tcn_prediction: torch.Tensor,
        anomaly_score: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if tcn_prediction.dim() == 0:
            tcn_prediction = tcn_prediction.unsqueeze(0)
        
        anomaly_tensor = torch.tensor([[anomaly_score]], dtype=torch.float32).to(self.device)
        
        gate_value = self.anomaly_gate(anomaly_tensor).item()
        
        self.anomaly_history.append(gate_value)
        
        debug_info = {
            'anomaly_score': anomaly_score,
            'gate_value': gate_value,
            'threshold': self.anomaly_threshold,
            'memory_size': len(self.memory_buffer),
            'is_correction_active': gate_value > 0.1
        }
        
        if gate_value < 0.1 or not self.is_initialized:
            self.update_memory(tcn_prediction)
            return tcn_prediction, debug_info
        
        history = self.memory_buffer.get_recent(min(20, len(self.memory_buffer))).to(self.device)
        
        if history.size(0) < 20:
            padding = torch.zeros(20 - history.size(0), self.trajectory_dim).to(self.device)
            history = torch.cat([padding, history], dim=0)
        
        history = history.unsqueeze(0)
        tcn_pred_device = tcn_prediction.to(self.device)
        
        corrected = self.lstm_corrector(history, tcn_pred_device)
        
        blend_factor = gate_value * self.correction_weight
        
        final_correction = (1 - blend_factor) * tcn_pred_device + blend_factor * corrected
        
        self.update_memory(final_correction)
        
        debug_info['gate_value'] = gate_value
        debug_info['blend_factor'] = blend_factor
        debug_info['corrected_trajectory'] = corrected.cpu().detach().numpy().tolist()
        
        return final_correction.cpu(), debug_info


class NormalTrajMemoryWithAttention(NormalTrajMemory):
    def __init__(self, *args, attention_heads: int = 4, **kwargs):
        self.attention_heads = attention_heads
        
        trajectory_dim = kwargs.get('trajectory_dim', 3)
        if trajectory_dim % attention_heads != 0:
            padding = (attention_heads - trajectory_dim % attention_heads)
            self.pad_dim = trajectory_dim + padding
        else:
            self.pad_dim = trajectory_dim
            padding = 0
        
        super().__init__(*args, **kwargs)
        
        self.padding_layer = nn.Linear(trajectory_dim, self.pad_dim) if padding > 0 else None
        self.unpadding_layer = nn.Linear(self.pad_dim, trajectory_dim) if padding > 0 else None
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.pad_dim,
            num_heads=min(attention_heads, self.pad_dim),
            dropout=kwargs.get('dropout', 0.1)
        ).to(self.device)
    
    def correct_trajectory(self, tcn_prediction, anomaly_score):
        if tcn_prediction.dim() == 0:
            tcn_prediction = tcn_prediction.unsqueeze(0)
        
        anomaly_tensor = torch.tensor([[anomaly_score]], dtype=torch.float32).to(self.device)
        gate_value = self.anomaly_gate(anomaly_tensor).item()
        
        self.update_memory(tcn_prediction)
        
        if gate_value < 0.1 or not self.is_initialized or len(self.memory_buffer) < 5:
            return tcn_prediction, {'gate_value': gate_value, 'correction_type': 'none'}
        
        history = self.memory_buffer.get_all().to(self.device)
        
        if history.size(0) < 2:
            return tcn_prediction, {'gate_value': gate_value, 'correction_type': 'insufficient_data'}
        
        history_padded = history
        if self.padding_layer is not None:
            history_padded = self.padding_layer(history)
        
        history_seq = history_padded
        if history_seq.dim() == 2:
            history_seq = history_seq.unsqueeze(0)
        
        attn_output, attn_weights = self.multihead_attention(
            history_seq, history_seq, history_seq
        )
        
        weighted_history = attn_output.squeeze(0)
        
        if self.unpadding_layer is not None:
            weighted_history = self.unpadding_layer(weighted_history)
        
        expected_trajectory = weighted_history.mean(dim=0)
        
        correction = expected_trajectory - tcn_prediction.squeeze().to(self.device)
        
        blend_factor = gate_value * self.correction_weight
        
        final_correction = tcn_prediction.to(self.device) + blend_factor * correction
        
        debug_info = {
            'gate_value': gate_value,
            'blend_factor': blend_factor,
            'correction_type': 'attention',
            'attention_weights': attn_weights.cpu().detach().numpy()
        }
        
        return final_correction.cpu(), debug_info


def create_normal_traj_memory(
    trajectory_dim: int,
    memory_size: int = 100,
    hidden_size: int = 64,
    anomaly_threshold: float = 0.5,
    correction_weight: float = 0.7,
    use_attention: bool = False,
    device: Optional[str] = None
) -> NormalTrajMemory:
    if use_attention:
        return NormalTrajMemoryWithAttention(
            trajectory_dim=trajectory_dim,
            memory_size=memory_size,
            hidden_size=hidden_size,
            anomaly_threshold=anomaly_threshold,
            correction_weight=correction_weight,
            device=device
        )
    else:
        return NormalTrajMemory(
            trajectory_dim=trajectory_dim,
            memory_size=memory_size,
            hidden_size=hidden_size,
            anomaly_threshold=anomaly_threshold,
            correction_weight=correction_weight,
            device=device
        )


if __name__ == '__main__':
    print("=" * 60)
    print("NormalTrajMemory Module - Usage Example")
    print("=" * 60)
    
    trajectory_dim = 3
    memory = create_normal_traj_memory(
        trajectory_dim=trajectory_dim,
        memory_size=100,
        hidden_size=64,
        anomaly_threshold=0.5,
        correction_weight=0.7,
        device='cpu'
    )
    
    print(f"Trajectory dimension: {trajectory_dim}")
    print(f"Memory size: {memory.memory_size}")
    print(f"Device: {memory.device}")
    print("-" * 60)
    
    print("\n1. Simulating normal trajectory storage...")
    for i in range(20):
        normal_traj = torch.randn(trajectory_dim) * 0.1
        memory.update_memory(normal_traj)
    
    stats = memory.get_memory_statistics()
    print(f"   Memory statistics: {stats}")
    
    print("\n2. Testing trajectory correction with low anomaly score...")
    tcn_pred = torch.randn(trajectory_dim) * 0.5
    anomaly_score = 0.2
    corrected_traj, debug_info = memory.correct_trajectory(tcn_pred, anomaly_score)
    print(f"   TCN Prediction: {tcn_pred.numpy()}")
    print(f"   Anomaly Score: {anomaly_score}")
    print(f"   Gate Value: {debug_info['gate_value']:.4f}")
    print(f"   Correction Applied: {debug_info['is_correction_active']}")
    
    print("\n3. Testing trajectory correction with high anomaly score...")
    tcn_pred = torch.randn(trajectory_dim) * 2.0
    anomaly_score = 0.85
    corrected_traj, debug_info = memory.correct_trajectory(tcn_pred, anomaly_score)
    print(f"   TCN Prediction: {tcn_pred.numpy()}")
    print(f"   Anomaly Score: {anomaly_score}")
    print(f"   Gate Value: {debug_info['gate_value']:.4f}")
    print(f"   Correction Applied: {debug_info['is_correction_active']}")
    
    print("\n4. Testing parameter adjustment...")
    memory.set_anomaly_threshold(0.3)
    memory.set_correction_weight(0.5)
    print(f"   New anomaly threshold: {memory.anomaly_threshold}")
    print(f"   New correction weight: {memory.correction_weight}")
    
    print("\n5. Testing module reset...")
    memory.reset()
    stats = memory.get_memory_statistics()
    print(f"   Memory after reset: {stats}")
    
    print("\n6. Testing with attention-based correction...")
    memory_attn = create_normal_traj_memory(
        trajectory_dim=trajectory_dim,
        memory_size=100,
        use_attention=True,
        device='cpu'
    )
    
    for i in range(15):
        normal_traj = torch.randn(trajectory_dim) * 0.1
        memory_attn.update_memory(normal_traj)
    
    tcn_pred = torch.randn(trajectory_dim) * 1.5
    anomaly_score = 0.7
    corrected_traj, debug_info = memory_attn.correct_trajectory(tcn_pred, anomaly_score)
    print(f"   TCN Prediction: {tcn_pred.numpy()}")
    print(f"   Correction Type: {debug_info['correction_type']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)