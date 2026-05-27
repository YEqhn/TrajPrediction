import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TCN.tcn import TemporalConvNet
from TCN.prediction_hip_test.model import TCNRegressor
from TCN.prediction_hip_test.utils import load_dataset_data, data_generator, denormalize_error, denormalize_mae, denormalize_rmse
from TCN.normal_traj_memory import NormalTrajMemory
from TimesNet_Model.timesnet_anomaly_detector import TimesNetAnomalyDetector, AnomalyInjector, IMUDataLoader, train_timesnet_anomaly_detector


print("=" * 80)
print("Integrated TCN + TimesNet + NormalTrajMemory Training System")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print("=" * 80)


class TrainingMonitor:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = {
            'tcn_train_loss': [], 'tcn_val_loss': [], 'tcn_rmse': [],
            'timesnet_loss': [], 'anomaly_scores': [],
            'correction_count': [], 'correction_effects': [],
            'epoch_times': [], 'total_inference_time': []
        }
        self.start_time = time.time()
        os.makedirs(log_dir, exist_ok=True)
        
    def log_metric(self, module: str, name: str, value: float):
        key = f"{module}_{name}"
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({
            'value': value,
            'timestamp': time.time() - self.start_time
        })
    
    def log_epoch(self, epoch: int, metrics: Dict):
        self.metrics['epoch_times'].append(time.time() - self.start_time)
        for module, module_metrics in metrics.items():
            for name, value in module_metrics.items():
                self.log_metric(module, name, value)
    
    def save_logs(self, filename: str = None):
        if filename is None:
            filename = f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"[MONITOR] Logs saved to: {filepath}")
        return filepath


class IntegratedTrainingSystem:
    def __init__(
        self,
        data_path: str,
        tcn_save_path: str,
        timesnet_save_path: str,
        log_dir: str,
        use_cuda: bool = True,
        batch_size: int = 32,
        tcn_epochs: int = 50,
        timesnet_epochs: int = 20,
        lr: float = 1e-3,
        window_seconds: float = 1.0,
        predict_ms: int = 100,
        sample_rate: int = 200,
        anomaly_threshold: float = 0.5,
        correction_weight: float = 0.7,
        memory_size: int = 100,
        device: str = None
    ):
        self.data_path = data_path
        self.tcn_save_path = tcn_save_path
        self.timesnet_save_path = timesnet_save_path
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.tcn_epochs = tcn_epochs
        self.timesnet_epochs = timesnet_epochs
        self.lr = lr
        self.window_seconds = window_seconds
        self.predict_ms = predict_ms
        self.sample_rate = sample_rate
        
        self.device = device if device else ('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
        print(f"[INIT] Using device: {self.device}")
        
        self.monitor = TrainingMonitor(log_dir)
        
        self.tcn_model = None
        self.timesnet_model = None
        self.normal_traj_memory = None
        self.anomaly_injector = AnomalyInjector()
        
        self.norm_params = None
        self.best_tcn_loss = float('inf')
        self.training_history = []
        
        os.makedirs(os.path.dirname(tcn_save_path) if os.path.dirname(tcn_save_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(timesnet_save_path) if os.path.dirname(timesnet_save_path) else '.', exist_ok=True)
        
    def _build_tcn_model(self, input_size: int, output_size: int, num_channels: List[int], kernel_size: int, dropout: float):
        self.tcn_model = TCNRegressor(input_size, output_size, num_channels, kernel_size, dropout)
        self.tcn_model = self.tcn_model.to(self.device)
        return self.tcn_model
    
    def _build_timesnet_model(self, input_channels: int, seq_len: int):
        self.timesnet_model = TimesNetAnomalyDetector(
            input_channels=input_channels,
            seq_len=seq_len,
            d_model=32,
            d_ff=128,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            dropout=0.1
        )
        self.timesnet_model = self.timesnet_model.to(self.device)
        return self.timesnet_model
    
    def _build_normal_traj_memory(self, trajectory_dim: int):
        self.normal_traj_memory = NormalTrajMemory(
            trajectory_dim=trajectory_dim,
            memory_size=100,
            hidden_size=64,
            lstm_layers=2,
            dropout=0.1,
            anomaly_threshold=0.5,
            correction_weight=0.7,
            device=self.device
        )
        return self.normal_traj_memory
    
    def load_data(self) -> Tuple:
        print("\n" + "=" * 60)
        print("Loading TCN Training Data")
        print("=" * 60)
        
        train_dataset, test_dataset, input_size, output_size, norm_params = load_dataset_data(
            self.data_path,
            train_ratio=0.8,
            window_seconds=self.window_seconds,
            predict_ms=self.predict_ms,
            sample_rate=self.sample_rate
        )
        
        self.norm_params = norm_params
        print(f"[DATA] Train samples: {len(train_dataset)}")
        print(f"[DATA] Test samples: {len(test_dataset)}")
        print(f"[DATA] Input size: {input_size}, Output size: {output_size}")
        
        return train_dataset, test_dataset, input_size, output_size
    
    def train_tcn(self, train_dataset: List, test_dataset: List, input_size: int, output_size: int):
        print("\n" + "=" * 60)
        print("Training TCN Module")
        print("=" * 60)
        
        num_channels = [64] * 6
        kernel_size = 5
        dropout = 0.2
        
        model = self._build_tcn_model(input_size, output_size, num_channels, kernel_size, dropout)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[TCN] Total parameters: {total_params:,}")
        
        def calculate_metrics(predictions, targets):
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).mean().item()
            rmse = np.sqrt(mse)
            return mse, mae, rmse
        
        best_loss = float('inf')
        
        for epoch in range(1, self.tcn_epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0
            train_steps = 0
            
            for batch_idx, (data, target) in enumerate(data_generator(train_dataset, self.batch_size)):
                if self.device == 'cuda':
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                
                optimizer.zero_grad()
                output = model(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / max(train_steps, 1)
            
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for data, target in data_generator(test_dataset, self.batch_size, shuffle=False):
                    if self.device == 'cuda':
                        data, target = data.cuda(), target.cuda()
                    output = model(data)
                    val_loss += F.mse_loss(output, target).item()
                    val_steps += 1
            
            avg_val_loss = val_loss / max(val_steps, 1)
            epoch_time = time.time() - epoch_start
            
            self.monitor.log_metric('tcn', 'train_loss', avg_train_loss)
            self.monitor.log_metric('tcn', 'val_loss', avg_val_loss)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'input_size': input_size,
                        'output_size': output_size,
                        'num_channels': num_channels,
                        'kernel_size': kernel_size
                    }
                }, self.tcn_save_path)
            
            if epoch % 10 == 0:
                lr = self.lr / (2 ** (epoch // 10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            print(f"[TCN] Epoch {epoch}/{self.tcn_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.1f}s")
        
        print(f"[TCN] Best model saved to: {self.tcn_save_path}")
        return model
    
    def train_timesnet(self, imu_data_path: str):
        print("\n" + "=" * 60)
        print("Training TimesNet Anomaly Detector")
        print("=" * 60)
        
        self.timesnet_model = train_timesnet_anomaly_detector(
            data_path=imu_data_path,
            epochs=self.timesnet_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            save_path=self.timesnet_save_path
        )
        
        print(f"[TimesNet] Model saved to: {self.timesnet_save_path}")
        return self.timesnet_model
    
    def run_integrated_inference(self, test_dataset: List, input_size: int, output_size: int):
        print("\n" + "=" * 60)
        print("Running Integrated Inference with Anomaly Detection")
        print("=" * 60)
        
        if self.tcn_model is None or self.timesnet_model is None or self.normal_traj_memory is None:
            raise RuntimeError("Models not initialized. Run training first.")
        
        self.tcn_model.eval()
        self.timesnet_model.eval()
        
        trajectory_dim = output_size
        self._build_normal_traj_memory(trajectory_dim)
        
        inference_times = []
        anomaly_scores = []
        correction_count = 0
        trajectory_errors = []
        
        print("\n[INFERENCE] Processing test samples...")
        
        for idx, (data, target) in enumerate(test_dataset[:100]):
            start_time = time.time()
            
            data_unsqueezed = data.unsqueeze(0)
            if self.device == 'cuda':
                data_unsqueezed = data_unsqueezed.cuda()
            
            tcn_pred = self.tcn_model(data_unsqueezed)
            
            imu_seq_len = data.shape[-1]
            imu_features = data.shape[0]
            imu_batch = data_unsqueezed.permute(0, 2, 1)
            
            anomaly_scores_batch, _ = self.timesnet_model.compute_anomaly_score(imu_batch)
            anomaly_score = anomaly_scores_batch.mean().item()
            anomaly_scores.append(anomaly_score)
            
            corrected_pred, debug_info = self.normal_traj_memory.correct_trajectory(
                tcn_pred.squeeze(0),
                anomaly_score
            )
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            if debug_info.get('is_correction_active', False):
                correction_count += 1
            
            error = F.mse_loss(corrected_pred.cpu(), target).item()
            trajectory_errors.append(error)
            
            if idx % 20 == 0:
                print(f"  Sample {idx:3d} | Anomaly: {anomaly_score:.4f} | Correction: {debug_info.get('is_correction_active', False)} | Time: {inference_time*1000:.1f}ms")
        
        avg_inference_time = np.mean(inference_times) * 1000
        avg_anomaly_score = np.mean(anomaly_scores)
        correction_rate = correction_count / len(inference_times) * 100
        
        print(f"\n[RESULTS]")
        print(f"  Average inference time: {avg_inference_time:.2f}ms")
        print(f"  Average anomaly score: {avg_anomaly_score:.4f}")
        print(f"  Correction rate: {correction_rate:.1f}%")
        print(f"  Average trajectory error: {np.mean(trajectory_errors):.6f}")
        
        return {
            'avg_inference_time_ms': avg_inference_time,
            'avg_anomaly_score': avg_anomaly_score,
            'correction_rate': correction_rate,
            'avg_trajectory_error': np.mean(trajectory_errors)
        }
    
    def run_anomaly_test(self, imu_data_path: str, num_test_samples: int = 50):
        print("\n" + "=" * 60)
        print("Running Anomaly Injection Tests")
        print("=" * 60)
        
        imu_loader = IMUDataLoader(imu_data_path)
        data, labels = imu_loader.prepare_training_data()
        
        if len(data) == 0:
            print("[WARN] No IMU data found for anomaly testing")
            return {}
        
        print(f"[TEST] Testing with {min(num_test_samples, len(data))} samples")
        
        self.timesnet_model.eval()
        
        normal_detected = 0
        anomaly_detected = 0
        test_results = {
            'normal': {'detected': 0, 'total': 0},
            'sensor_noise': {'detected': 0, 'total': 0},
            'data_dropout': {'detected': 0, 'total': 0},
            'spike_anomaly': {'detected': 0, 'total': 0},
            'sensor_drift': {'detected': 0, 'total': 0},
            'data_freeze': {'detected': 0, 'total': 0}
        }
        
        for idx in range(min(num_test_samples, len(data))):
            sample = data[idx:idx+1].to(self.device)
            
            injected_sample, anomaly_type = self.anomaly_injector.inject_random_anomaly_type(sample)
            
            scores, _ = self.timesnet_model.compute_anomaly_score(injected_sample)
            avg_score = scores.mean().item()
            
            is_detected = self.timesnet_model.is_anomalous(scores, threshold_multiplier=1.5)
            
            test_results[anomaly_type]['total'] += 1
            if anomaly_type == 'normal':
                if not is_detected:
                    normal_detected += 1
                test_results['normal']['detected'] += 0 if is_detected else 1
            else:
                if is_detected:
                    anomaly_detected += 1
                    test_results[anomaly_type]['detected'] += 1
        
        print(f"\n[ANOMALY TEST RESULTS]")
        print(f"  Normal samples correctly classified: {normal_detected}/{test_results['normal']['total']}")
        
        for anomaly_type in ['sensor_noise', 'data_dropout', 'spike_anomaly', 'sensor_drift', 'data_freeze']:
            if test_results[anomaly_type]['total'] > 0:
                detected = test_results[anomaly_type]['detected']
                total = test_results[anomaly_type]['total']
                rate = detected / total * 100
                print(f"  {anomaly_type}: {detected}/{total} ({rate:.1f}%)")
        
        total_anomalies = sum(test_results[t]['total'] for t in test_results if t != 'normal')
        total_detected = sum(test_results[t]['detected'] for t in test_results if t != 'normal')
        if total_anomalies > 0:
            print(f"  Overall anomaly detection rate: {total_detected}/{total_anomalies} ({total_detected/total_anomalies*100:.1f}%)")
        
        return test_results
    
    def save_checkpoint(self, filepath: str):
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'tcn_model_path': self.tcn_save_path,
            'timesnet_model_path': self.timesnet_save_path,
            'training_history': self.training_history,
            'norm_params': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in self.norm_params.items()} if self.norm_params else None
        }
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"[CHECKPOINT] Saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Integrated TCN + TimesNet + NormalTrajMemory Training')
    
    parser.add_argument('--data_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/data_processed',
                       help='Path to TCN data folder')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--tcn_epochs', type=int, default=50, help='TCN training epochs')
    parser.add_argument('--timesnet_epochs', type=int, default=20, help='TimesNet training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--window_seconds', type=float, default=1.0, help='Input window duration')
    parser.add_argument('--predict_ms', type=int, default=100, help='Prediction horizon in ms')
    parser.add_argument('--sample_rate', type=int, default=200, help='Sample rate in Hz')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, help='Anomaly threshold')
    parser.add_argument('--log_dir', type=str, default='/home/yeqhn/Desktop/TCN/logs', help='Log directory')
    parser.add_argument('--tcn_save_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/models/tcn_model.pt',
                       help='TCN model save path')
    parser.add_argument('--timesnet_save_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/models/timesnet_anomaly_detector.pt',
                       help='TimesNet model save path')
    parser.add_argument('--skip_tcn', action='store_true', help='Skip TCN training')
    parser.add_argument('--skip_timesnet', action='store_true', help='Skip TimesNet training')
    parser.add_argument('--run_tests', action='store_true', help='Run anomaly tests')
    
    args = parser.parse_args()
    
    os.makedirs('/home/yeqhn/Desktop/TCN/models', exist_ok=True)
    
    system = IntegratedTrainingSystem(
        data_path=args.data_path,
        tcn_save_path=args.tcn_save_path,
        timesnet_save_path=args.timesnet_save_path,
        log_dir=args.log_dir,
        use_cuda=args.cuda,
        batch_size=args.batch_size,
        tcn_epochs=args.tcn_epochs,
        timesnet_epochs=args.timesnet_epochs,
        lr=args.lr,
        window_seconds=args.window_seconds,
        predict_ms=args.predict_ms,
        sample_rate=args.sample_rate,
        anomaly_threshold=args.anomaly_threshold
    )
    
    print("\n" + "=" * 80)
    print("Starting Integrated Training Pipeline")
    print("=" * 80)
    
    train_dataset, test_dataset, input_size, output_size = system.load_data()
    
    if not args.skip_tcn:
        tcn_model = system.train_tcn(train_dataset, test_dataset, input_size, output_size)
    else:
        print("[SKIP] TCN training skipped")
        if os.path.exists(args.tcn_save_path):
            checkpoint = torch.load(args.tcn_save_path)
            system.tcn_model = TCNRegressor(
                checkpoint['config']['input_size'],
                checkpoint['config']['output_size'],
                checkpoint['config']['num_channels'],
                checkpoint['config']['kernel_size']
            )
            system.tcn_model.load_state_dict(checkpoint['model_state_dict'])
            system.tcn_model = system.tcn_model.to(system.device)
            print(f"[LOAD] TCN model loaded from: {args.tcn_save_path}")
    
    if not args.skip_timesnet:
        timesnet_model = system.train_timesnet(args.data_path)
    else:
        print("[SKIP] TimesNet training skipped")
        if os.path.exists(args.timesnet_save_path):
            checkpoint = torch.load(args.timesnet_save_path)
            system.timesnet_model = TimesNetAnomalyDetector(
                input_channels=checkpoint['config']['input_channels'],
                seq_len=checkpoint['config']['seq_len'],
                d_model=checkpoint['config']['d_model'],
                d_ff=checkpoint['config']['d_ff'],
                e_layers=checkpoint['config']['e_layers']
            )
            system.timesnet_model.load_state_dict(checkpoint['model_state_dict'])
            system.timesnet_model.baseline_threshold = checkpoint.get('baseline_threshold')
            system.timesnet_model.baseline_mean = checkpoint.get('baseline_mean')
            system.timesnet_model.baseline_std = checkpoint.get('baseline_std')
            system.timesnet_model = system.timesnet_model.to(system.device)
            print(f"[LOAD] TimesNet model loaded from: {args.timesnet_save_path}")
    
    print("\n" + "=" * 80)
    print("Running Integrated Inference")
    print("=" * 80)
    
    inference_results = system.run_integrated_inference(test_dataset, input_size, output_size)
    
    if args.run_tests:
        print("\n" + "=" * 80)
        print("Running Anomaly Injection Tests")
        print("=" * 80)
        
        anomaly_results = system.run_anomaly_test(args.data_path)
    
    system.monitor.save_logs()
    system.save_checkpoint('/home/yeqhn/Desktop/TCN/models/training_checkpoint.json')
    
    print("\n" + "=" * 80)
    print("Training Pipeline Completed!")
    print("=" * 80)
    print(f"TCN Model: {args.tcn_save_path}")
    print(f"TimesNet Model: {args.timesnet_save_path}")
    print(f"Logs: {args.log_dir}")


if __name__ == "__main__":
    main()