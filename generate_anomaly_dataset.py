import numpy as np
import pandas as pd
import os
from typing import List, Tuple

class AnomalyIMUDataGenerator:
    def __init__(
        self,
        source_imu_path: str,
        output_path: str,
        anomaly_ratio: float = 0.12
    ):
        self.source_imu_path = source_imu_path
        self.output_path = output_path
        self.anomaly_ratio = anomaly_ratio
        self.anomaly_types = [
            'sensor_noise',
            'data_dropout',
            'spike_anomaly',
            'sensor_drift',
            'data_freeze'
        ]
        self.anomaly_probs = [0.25, 0.20, 0.20, 0.20, 0.15]

    def _inject_sensor_noise(self, data: np.ndarray, intensity: float = 2.0) -> np.ndarray:
        noise = np.random.randn(*data.shape) * intensity
        return data + noise

    def _inject_data_dropout(self, data: np.ndarray, drop_ratio: float = 0.25) -> np.ndarray:
        mask = np.random.rand(*data.shape) > drop_ratio
        return data * mask

    def _inject_spike_anomaly(self, data: np.ndarray, spike_ratio: float = 0.12) -> np.ndarray:
        result = data.copy()
        T, N = data.shape
        num_spikes = int(T * spike_ratio)
        spike_indices = np.random.choice(T, size=num_spikes, replace=False)
        for idx in spike_indices:
            for n in range(N):
                result[idx, n] += np.random.randn() * 5.0
        return result

    def _inject_sensor_drift(self, data: np.ndarray, drift_rate: float = 0.015) -> np.ndarray:
        result = data.copy()
        T, N = data.shape
        t = np.arange(T, dtype=np.float32)
        for n in range(N):
            std = np.std(data[:, n]) if np.std(data[:, n]) > 0 else 1.0
            drift = t * drift_rate * np.random.randn() * std
            result[:, n] += drift
        return result

    def _inject_data_freeze(self, data: np.ndarray, freeze_ratio: float = 0.18) -> np.ndarray:
        result = data.copy()
        T, N = data.shape
        freeze_start = int(T * 0.35)
        freeze_end = freeze_start + int(T * freeze_ratio)
        freeze_value = data[freeze_start - 1, :].copy()
        for t in range(freeze_start, min(freeze_end, T)):
            result[t, :] = freeze_value + np.random.randn(N) * 0.1
        return result

    def _apply_anomaly(self, data: np.ndarray, anomaly_type: str) -> np.ndarray:
        if anomaly_type == 'sensor_noise':
            return self._inject_sensor_noise(data, intensity=np.random.uniform(1.5, 3.0))
        elif anomaly_type == 'data_dropout':
            return self._inject_data_dropout(data, drop_ratio=np.random.uniform(0.15, 0.35))
        elif anomaly_type == 'spike_anomaly':
            return self._inject_spike_anomaly(data, spike_ratio=np.random.uniform(0.08, 0.20))
        elif anomaly_type == 'sensor_drift':
            return self._inject_sensor_drift(data, drift_rate=np.random.uniform(0.008, 0.025))
        elif anomaly_type == 'data_freeze':
            return self._inject_data_freeze(data, freeze_ratio=np.random.uniform(0.12, 0.25))
        return data

    def generate_anomaly_data(self, window_size: int = 100) -> Tuple[List[np.ndarray], List[str]]:
        imu_files = sorted([f for f in os.listdir(self.source_imu_path) if f.endswith('.csv')])
        if not imu_files:
            raise FileNotFoundError(f"No CSV files found in {self.source_imu_path}")

        print(f"Found {len(imu_files)} IMU files")

        all_anomaly_data = []
        all_anomaly_labels = []
        total_samples = 0
        anomaly_samples = 0

        for filename in imu_files:
            filepath = os.path.join(self.source_imu_path, filename)
            df = pd.read_csv(filepath)
            header = df.columns[0]
            data = df.iloc[:, 1:].values

            num_windows = len(data) // window_size
            samples_in_file = 0

            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                window_data = data[start_idx:end_idx]

                total_samples += 1

                if np.random.rand() < self.anomaly_ratio:
                    anomaly_type = np.random.choice(self.anomaly_types, p=self.anomaly_probs)
                    window_data = self._apply_anomaly(window_data, anomaly_type)
                    anomaly_samples += 1
                    samples_in_file += 1
                    all_anomaly_labels.append(f"{filename}_{i}_{anomaly_type}")
                else:
                    all_anomaly_labels.append(f"{filename}_{i}_normal")

                all_anomaly_data.append(window_data)

            print(f"  {filename}: {samples_in_file} anomaly windows out of {num_windows} total")

        print(f"\n[Generated] Total samples: {total_samples}, Anomaly samples: {anomaly_samples} ({anomaly_samples/total_samples*100:.1f}%)")
        return all_anomaly_data, all_anomaly_labels

    def save_anomaly_dataset(self, anomaly_data: List[np.ndarray], labels: List[str]):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'data'), exist_ok=True)

        df_all = []
        for i, (data, label) in enumerate(zip(anomaly_data, labels)):
            for j in range(len(data)):
                row = data[j].tolist()
                row.insert(0, f"{label}_{j}")
                df_all.append(row)

        column_names = ['Header',
                        'foot_Accel_X', 'foot_Accel_Y', 'foot_Accel_Z',
                        'foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z',
                        'shank_Accel_X', 'shank_Accel_Y', 'shank_Accel_Z',
                        'shank_Gyro_X', 'shank_Gyro_Y', 'shank_Gyro_Z',
                        'thigh_Accel_X', 'thigh_Accel_Y', 'thigh_Accel_Z',
                        'thigh_Gyro_X', 'thigh_Gyro_Y', 'thigh_Gyro_Z',
                        'trunk_Accel_X', 'trunk_Accel_Y', 'trunk_Accel_Z',
                        'trunk_Gyro_X', 'trunk_Gyro_Y', 'trunk_Gyro_Z']

        df = pd.DataFrame(df_all, columns=column_names)

        output_file = os.path.join(self.output_path, 'anomaly_imu_dataset.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved anomaly dataset to: {output_file}")
        print(f"Dataset shape: {df.shape}")

        label_file = os.path.join(self.output_path, 'anomaly_labels.csv')
        label_df = pd.DataFrame({
            'sample_id': [labels[i].rsplit('_', 1)[0] for i in range(len(labels))],
            'anomaly_type': [labels[i].rsplit('_', 1)[1] if '_' in labels[i] else 'normal' for i in range(len(labels))]
        })
        label_df.to_csv(label_file, index=False)
        print(f"Saved labels to: {label_file}")

        return output_file

    def generate_split_dataset(self, train_ratio: float = 0.8):
        print("=" * 60)
        print("Generating Anomaly IMU Dataset")
        print("=" * 60)

        anomaly_data, labels = self.generate_anomaly_data(window_size=100)

        indices = np.random.permutation(len(anomaly_data))
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        os.makedirs(os.path.join(self.output_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'test'), exist_ok=True)

        self._save_split_data(
            [anomaly_data[i] for i in train_indices],
            [labels[i] for i in train_indices],
            os.path.join(self.output_path, 'train')
        )

        self._save_split_data(
            [anomaly_data[i] for i in test_indices],
            [labels[i] for i in test_indices],
            os.path.join(self.output_path, 'test')
        )

        print("=" * 60)
        print("Split dataset generation completed")
        print(f"Train samples: {len(train_indices)}")
        print(f"Test samples: {len(test_indices)}")
        print("=" * 60)

    def _save_split_data(self, data_list: List[np.ndarray], labels: List[str], output_dir: str):
        column_names = ['Header',
                        'foot_Accel_X', 'foot_Accel_Y', 'foot_Accel_Z',
                        'foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z',
                        'shank_Accel_X', 'shank_Accel_Y', 'shank_Accel_Z',
                        'shank_Gyro_X', 'shank_Gyro_Y', 'shank_Gyro_Z',
                        'thigh_Accel_X', 'thigh_Accel_Y', 'thigh_Accel_Z',
                        'thigh_Gyro_X', 'thigh_Gyro_Y', 'thigh_Gyro_Z',
                        'trunk_Accel_X', 'trunk_Accel_Y', 'trunk_Accel_Z',
                        'trunk_Gyro_X', 'trunk_Gyro_Y', 'trunk_Gyro_Z']

        rows = []
        for data, label in zip(data_list, labels):
            for j in range(len(data)):
                row = data[j].tolist()
                row.insert(0, f"{label}_{j}")
                rows.append(row)

        df = pd.DataFrame(rows, columns=column_names)
        output_file = os.path.join(output_dir, 'anomaly_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(data_list)} windows, {len(rows)} rows)")


def main():
    source_path = '/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/imu'
    output_path = '/home/yeqhn/Desktop/TCN/anomaly_data'

    generator = AnomalyIMUDataGenerator(
        source_imu_path=source_path,
        output_path=output_path,
        anomaly_ratio=0.12
    )

    generator.generate_split_dataset(train_ratio=0.8)

    print("\n" + "=" * 60)
    print("Anomaly dataset generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()