import torch
import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TimesNet_Model.timesnet_anomaly_detector import AnomalyInjector, TimesNetAnomalyDetector
from TCN.tcn import TemporalConvNet
from TCN.prediction_hip_test.model import TCNRegressor
from TCN.normal_traj_memory import NormalTrajMemory


class AnomalyIMUDataTester:
    def __init__(
        self,
        tcn_model_path: str,
        timesnet_model_path: str,
        norm_params_path: str,
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tcn_model = None
        self.timesnet_model = None
        self.normal_traj_memory = None
        self.norm_params = None
        self.anomaly_injector = AnomalyInjector()
        
        self.test_results = []
        
    def load_models(self, tcn_model_path: str, timesnet_model_path: str, norm_params_path: str):
        print("=" * 80)
        print("Loading Models")
        print("=" * 80)
        
        print(f"[LOAD] TCN model from: {tcn_model_path}")
        tcn_checkpoint = torch.load(tcn_model_path, map_location=self.device)
        self.tcn_model = TCNRegressor(
            tcn_checkpoint['config']['input_size'],
            tcn_checkpoint['config']['output_size'],
            tcn_checkpoint['config']['num_channels'],
            tcn_checkpoint['config']['kernel_size']
        )
        self.tcn_model.load_state_dict(tcn_checkpoint['model_state_dict'])
        self.tcn_model = self.tcn_model.to(self.device)
        self.tcn_model.eval()
        print(f"  TCN model loaded successfully")
        
        print(f"[LOAD] TimesNet model from: {timesnet_model_path}")
        timesnet_checkpoint = torch.load(timesnet_model_path, map_location=self.device)
        self.timesnet_model = TimesNetAnomalyDetector(
            input_channels=timesnet_checkpoint['config']['input_channels'],
            seq_len=timesnet_checkpoint['config']['seq_len'],
            d_model=timesnet_checkpoint['config']['d_model'],
            d_ff=timesnet_checkpoint['config']['d_ff'],
            e_layers=timesnet_checkpoint['config']['e_layers']
        )
        self.timesnet_model.load_state_dict(timesnet_checkpoint['model_state_dict'])
        self.timesnet_model.baseline_threshold = timesnet_checkpoint.get('baseline_threshold')
        self.timesnet_model.baseline_mean = timesnet_checkpoint.get('baseline_mean')
        self.timesnet_model.baseline_std = timesnet_checkpoint.get('baseline_std')
        self.timesnet_model = self.timesnet_model.to(self.device)
        self.timesnet_model.eval()
        print(f"  TimesNet model loaded successfully")
        
        print(f"[LOAD] Normalization params from: {norm_params_path}")
        self.norm_params = torch.load(norm_params_path, map_location='cpu')
        print(f"  Normalization params loaded successfully")
        
        output_size = tcn_checkpoint['config']['output_size']
        self.normal_traj_memory = NormalTrajMemory(
            trajectory_dim=output_size,
            memory_size=100,
            hidden_size=64,
            lstm_layers=2,
            dropout=0.1,
            anomaly_threshold=0.5,
            correction_weight=0.7,
            device=self.device
        )
        print(f"  NormalTrajMemory initialized")
        
        print("=" * 80)
    
    def _denormalize_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        output_mean = torch.tensor(self.norm_params['output_std'], dtype=torch.float32)
        output_std = torch.tensor(self.norm_params['output_std'], dtype=torch.float32)
        return trajectory * output_std + output_mean
    
    def test_single_sample(
        self,
        imu_data: torch.Tensor,
        ground_truth: torch.Tensor,
        inject_anomaly: bool = False,
        anomaly_type: str = None
    ) -> Dict:
        result = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_type': 'normal',
            'is_anomaly_injected': inject_anomaly
        }
        
        start_time = time.time()
        
        processed_data = imu_data
        if inject_anomaly:
            if anomaly_type:
                if anomaly_type == 'sensor_noise':
                    processed_data = self.anomaly_injector.inject_sensor_noise(imu_data.unsqueeze(0), intensity=2.0).squeeze(0)
                elif anomaly_type == 'data_dropout':
                    processed_data = self.anomaly_injector.inject_data_dropout(imu_data.unsqueeze(0), drop_ratio=0.3).squeeze(0)
                elif anomaly_type == 'spike_anomaly':
                    processed_data = self.anomaly_injector.inject_spike_anomaly(imu_data.unsqueeze(0), spike_ratio=0.15).squeeze(0)
                elif anomaly_type == 'sensor_drift':
                    processed_data = self.anomaly_injector.inject_drift(imu_data.unsqueeze(0), drift_rate=0.015).squeeze(0)
                elif anomaly_type == 'data_freeze':
                    processed_data = self.anomaly_injector.inject_freeze(imu_data.unsqueeze(0), freeze_ratio=0.2).squeeze(0)
                else:
                    processed_data = imu_data
            else:
                processed_data, detected_type = self.anomaly_injector.inject_random_anomaly_type(imu_data.unsqueeze(0))
                processed_data = processed_data.squeeze(0)
                anomaly_type = detected_type
            
            result['anomaly_type'] = anomaly_type
        
        tcn_input = processed_data.unsqueeze(0)
        if self.device == 'cuda':
            tcn_input = tcn_input.cuda()
        
        with torch.no_grad():
            tcn_pred = self.tcn_model(tcn_input)
        
        imu_seq = tcn_input.permute(0, 2, 1)
        anomaly_scores, _ = self.timesnet_model.compute_anomaly_score(imu_seq)
        anomaly_score = anomaly_scores.mean().item()
        result['anomaly_score'] = anomaly_score
        result['is_anomaly_detected'] = self.timesnet_model.is_anomalous(anomaly_scores, threshold_multiplier=1.5)
        
        corrected_pred, debug_info = self.normal_traj_memory.correct_trajectory(
            tcn_pred.squeeze(0),
            anomaly_score
        )
        
        inference_time = time.time() - start_time
        result['inference_time_ms'] = inference_time * 1000
        result['correction_active'] = debug_info.get('is_correction_active', False)
        
        mse_before = torch.nn.functional.mse_loss(tcn_pred.squeeze(0), ground_truth).item()
        mse_after = torch.nn.functional.mse_loss(corrected_pred.cpu(), ground_truth).item()
        
        if self.norm_params is not None:
            output_std = np.mean(self.norm_params['output_std'])
            result['mse_before_deg2'] = mse_before * (output_std ** 2)
            result['mse_after_deg2'] = mse_after * (output_std ** 2)
            result['rmse_before_deg'] = np.sqrt(mse_before) * output_std
            result['rmse_after_deg'] = np.sqrt(mse_after) * output_std
        else:
            result['mse_before'] = mse_before
            result['mse_after'] = mse_after
            result['rmse_before'] = np.sqrt(mse_before)
            result['rmse_after'] = np.sqrt(mse_after)
        
        result['improvement'] = (mse_before - mse_after) / mse_before * 100 if mse_before > 0 else 0
        
        return result
    
    def run_comparative_test(
        self,
        test_dataset: List,
        anomaly_types: List[str],
        samples_per_type: int = 20
    ) -> Dict:
        print("\n" + "=" * 80)
        print("Running Comparative Anomaly Tests")
        print("=" * 80)
        
        all_results = {
            'normal': [],
            'sensor_noise': [],
            'data_dropout': [],
            'spike_anomaly': [],
            'sensor_drift': [],
            'data_freeze': []
        }
        
        normal_samples = []
        for i, (imu_data, ground_truth) in enumerate(test_dataset):
            if len(normal_samples) < samples_per_type:
                normal_samples.append((imu_data, ground_truth))
        
        print(f"[TEST] Testing {samples_per_type} normal samples...")
        for imu_data, ground_truth in normal_samples[:samples_per_type]:
            result = self.test_single_sample(imu_data, ground_truth, inject_anomaly=False)
            all_results['normal'].append(result)
        
        for anomaly_type in anomaly_types:
            print(f"[TEST] Testing {samples_per_type} samples with {anomaly_type}...")
            type_results = []
            
            for i, (imu_data, ground_truth) in enumerate(normal_samples):
                if len(type_results) >= samples_per_type:
                    break
                result = self.test_single_sample(
                    imu_data, ground_truth, 
                    inject_anomaly=True, 
                    anomaly_type=anomaly_type
                )
                type_results.append(result)
            
            all_results[anomaly_type] = type_results
        
        summary = {}
        print("\n" + "=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        
        print(f"\n{'Anomaly Type':<20} {'Count':>8} {'Avg Anomaly Score':>18} {'Detection Rate':>15} {'Avg RMSE Before':>15} {'Avg RMSE After':>15} {'Improvement':>12}")
        print("-" * 110)
        
        for anomaly_type, results in all_results.items():
            if len(results) == 0:
                continue
            
            count = len(results)
            avg_score = np.mean([r['anomaly_score'] for r in results])
            detection_rate = np.mean([r['is_anomaly_detected'] for r in results]) * 100
            avg_rmse_before = np.mean([r.get('rmse_before_deg', r.get('rmse_before', 0)) for r in results])
            avg_rmse_after = np.mean([r.get('rmse_after_deg', r.get('rmse_after', 0)) for r in results])
            avg_improvement = np.mean([r['improvement'] for r in results])
            
            summary[anomaly_type] = {
                'count': count,
                'avg_anomaly_score': float(avg_score),
                'detection_rate': float(detection_rate),
                'avg_rmse_before_deg': float(avg_rmse_before),
                'avg_rmse_after_deg': float(avg_rmse_after),
                'avg_improvement_pct': float(avg_improvement),
                'avg_inference_time_ms': float(np.mean([r['inference_time_ms'] for r in results]))
            }
            
            print(f"{anomaly_type:<20} {count:>8} {avg_score:>18.4f} {detection_rate:>14.1f}% {avg_rmse_before:>14.2f}° {avg_rmse_after:>14.2f}° {avg_improvement:>11.1f}%")
        
        print("=" * 80)
        
        return {
            'detailed_results': all_results,
            'summary': summary,
            'test_timestamp': datetime.now().isoformat()
        }
    
    def run_response_time_test(
        self,
        test_dataset: List,
        num_samples: int = 100
    ) -> Dict:
        print("\n" + "=" * 80)
        print("Testing Response Time Performance")
        print("=" * 80)
        
        response_times = []
        correction_times = []
        
        print(f"[TEST] Testing {num_samples} samples for response time...")
        
        for i, (imu_data, ground_truth) in enumerate(test_dataset[:num_samples]):
            start_time = time.time()
            
            tcn_input = imu_data.unsqueeze(0)
            if self.device == 'cuda':
                tcn_input = tcn_input.cuda()
            
            tcn_pred = self.tcn_model(tcn_input)
            
            imu_seq = tcn_input.permute(0, 2, 1)
            anomaly_scores, _ = self.timesnet_model.compute_anomaly_score(imu_seq)
            anomaly_score = anomaly_scores.mean().item()
            
            corrected_pred, _ = self.normal_traj_memory.correct_trajectory(
                tcn_pred.squeeze(0),
                anomaly_score
            )
            
            total_time = time.time() - start_time
            response_times.append(total_time * 1000)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
                sync_end = time.time()
                response_times[-1] = (sync_end - start_time) * 1000
        
        print(f"\n[RESPONSE TIME RESULTS]")
        print(f"  Total samples: {len(response_times)}")
        print(f"  Average: {np.mean(response_times):.2f}ms")
        print(f"  Median: {np.median(response_times):.2f}ms")
        print(f"  Min: {np.min(response_times):.2f}ms")
        print(f"  Max: {np.max(response_times):.2f}ms")
        print(f"  Std: {np.std(response_times):.2f}ms")
        print(f"  95th percentile: {np.percentile(response_times, 95):.2f}ms")
        print(f"  99th percentile: {np.percentile(response_times, 99):.2f}ms")
        
        if self.device == 'cuda':
            print(f"  Real-time capable: {'Yes' if np.percentile(response_times, 99) < 10 else 'No'} (target: <10ms)")
        else:
            print(f"  Real-time capable: {'Yes' if np.percentile(response_times, 99) < 50 else 'No'} (target: <50ms)")
        
        return {
            'avg_response_time_ms': float(np.mean(response_times)),
            'median_response_time_ms': float(np.median(response_times)),
            'min_response_time_ms': float(np.min(response_times)),
            'max_response_time_ms': float(np.max(response_times)),
            'std_response_time_ms': float(np.std(response_times)),
            'p95_response_time_ms': float(np.percentile(response_times, 95)),
            'p99_response_time_ms': float(np.percentile(response_times, 99)),
            'realtime_capable': bool(np.percentile(response_times, 99) < (10 if self.device == 'cuda' else 50))
        }
    
    def save_results(self, results: Dict, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[SAVE] Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Anomaly IMU Data Testing')
    parser.add_argument('--tcn_model_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/models/tcn_model.pt',
                       help='Path to TCN model')
    parser.add_argument('--timesnet_model_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/models/timesnet_anomaly_detector.pt',
                       help='Path to TimesNet model')
    parser.add_argument('--norm_params_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/TCN/prediction_hip_test/results/normalization.pt',
                       help='Path to normalization params')
    parser.add_argument('--data_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/data_processed',
                       help='Path to data folder')
    parser.add_argument('--samples_per_type', type=int, default=20, 
                       help='Number of samples per anomaly type')
    parser.add_argument('--output_path', type=str, 
                       default='/home/yeqhn/Desktop/TCN/logs/anomaly_test_results.json',
                       help='Output path for test results')
    
    args = parser.parse_args()
    
    from TCN.prediction_hip_test.utils import load_dataset_data
    
    print("Loading test data...")
    _, test_dataset, _, _ = load_dataset_data(
        args.data_path,
        train_ratio=0.8,
        window_seconds=1.0,
        predict_ms=100,
        sample_rate=200
    )
    print(f"Loaded {len(test_dataset)} test samples")
    
    tester = AnomalyIMUDataTester(
        tcn_model_path=args.tcn_model_path,
        timesnet_model_path=args.timesnet_model_path,
        norm_params_path=args.norm_params_path
    )
    
    if os.path.exists(args.tcn_model_path) and os.path.exists(args.timesnet_model_path):
        tester.load_models(args.tcn_model_path, args.timesnet_model_path, args.norm_params_path)
    else:
        print("[ERROR] Model files not found. Please run training first.")
        return
    
    anomaly_types = ['sensor_noise', 'data_dropout', 'spike_anomaly', 'sensor_drift', 'data_freeze']
    
    comparative_results = tester.run_comparative_test(
        test_dataset=test_dataset,
        anomaly_types=anomaly_types,
        samples_per_type=args.samples_per_type
    )
    
    response_time_results = tester.run_response_time_test(
        test_dataset=test_dataset,
        num_samples=100
    )
    
    final_results = {
        'comparative_test': comparative_results,
        'response_time_test': response_time_results,
        'config': {
            'samples_per_type': args.samples_per_type,
            'device': tester.device
        }
    }
    
    tester.save_results(final_results, args.output_path)
    
    print("\n" + "=" * 80)
    print("Testing Pipeline Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()