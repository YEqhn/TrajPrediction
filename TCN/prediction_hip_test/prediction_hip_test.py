import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from TCN.tcn import TemporalConvNet
from utils import load_dataset_data, data_generator, denormalize_predictions, denormalize_error, denormalize_mae, denormalize_rmse, calculate_rmse_from_mse
from model import TCNRegressor
import numpy as np
import argparse

print("=" * 80)
print("TCN Training - IMU-based Joint Angle Prediction")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
print("=" * 80)

parser = argparse.ArgumentParser(description='TCN for Joint Angle Prediction from IMU Data')
parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_true', help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (default: 0.2)')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=5, help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=6, help='# of levels (default: 6)')
parser.add_argument('--log_interval', type=int, default=50, help='report interval (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer (default: Adam)')
parser.add_argument('--nhid', type=int, default=64, help='hidden units per layer (default: 64)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--data_path', type=str, 
                    default='/home/yeqhn/Desktop/TCN/data_processed',
                    help='path to data_processed folder')
parser.add_argument('--save_path', type=str, 
                    default='/home/yeqhn/Desktop/TCN/TCN/prediction_hip_test/results/tcn_model.pt',
                    help='path to save model')
parser.add_argument('--norm_path', type=str, 
                    default='/home/yeqhn/Desktop/TCN/TCN/prediction_hip_test/results/normalization.pt',
                    help='path to save normalization params')
parser.add_argument('--log_dir', type=str,
                    default='/home/yeqhn/Desktop/TCN/logs',
                    help='path to save training logs')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='training data ratio (default: 0.8)')
parser.add_argument('--window_seconds', type=float, default=1.0,
                    help='input window duration in seconds (default: 1.0)')
parser.add_argument('--predict_ms', type=int, default=100,
                    help='prediction horizon in milliseconds (default: 100)')
parser.add_argument('--sample_rate', type=int, default=200,
                    help='sampling frequency in Hz (default: 200)')

args = parser.parse_args()

torch.manual_seed(args.seed)
use_cuda = args.cuda and torch.cuda.is_available()
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

batch_size = args.batch_size
epochs = args.epochs

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
os.makedirs(args.norm_path.rsplit('/', 1)[0], exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

log_file = os.path.join(args.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

print("\n" + "=" * 80)
print("Training Configuration")
print("=" * 80)
print(f"Data path: {args.data_path}")
print(f"Input window: {args.window_seconds}s ({int(args.window_seconds * args.sample_rate)} samples)")
print(f"Prediction horizon: {args.predict_ms}ms ({int(args.predict_ms * args.sample_rate / 1000)} samples)")
print(f"Sample rate: {args.sample_rate} Hz")
print(f"Train/Test ratio: {args.train_ratio}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {args.lr}")
print(f"Optimizer: {args.optim}")
print(f"Hidden units: {args.nhid}")
print(f"Levels: {args.levels}")
print(f"Kernel size: {args.ksize}")
print(f"Dropout: {args.dropout}")
print("=" * 80 + "\n")

train_dataset, test_dataset, input_size, output_size, norm_params = load_dataset_data(
    args.data_path,
    train_ratio=args.train_ratio,
    window_seconds=args.window_seconds,
    predict_ms=args.predict_ms,
    sample_rate=args.sample_rate
)

torch.save(norm_params, args.norm_path)
print(f"Normalization parameters saved to {args.norm_path}")

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize

model = TCNRegressor(input_size, output_size, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

print("\n" + "=" * 80)
print("Model Architecture")
print("=" * 80)
print(f"Input size: {input_size} (IMU features)")
print(f"Output size: {output_size} (joint angles)")
print(f"Sequence length: {int(args.window_seconds * args.sample_rate)}")
print(f"TCN channels: {channel_sizes}")
print(f"Kernel size: {kernel_size}")
print("=" * 80 + "\n")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}\n")

if use_cuda:
    model.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

best_test_loss = float('inf')
best_mse_denorm = None
best_rmse_denorm = None
train_history = []
val_history = []
def calculate_metrics(predictions, targets):
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).mean().item()
    rmse = np.sqrt(mse)
    return mse, mae, rmse


def train(ep):
    global best_test_loss
    train_loss = 0
    train_mse = 0
    train_mae = 0
    train_rmse = 0
    model.train()
    steps = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(data_generator(train_dataset, batch_size)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        
        train_loss += loss.item()
        mse, mae, rmse = calculate_metrics(output, target)
        train_mse += mse
        train_mae += mae
        train_rmse += rmse
        steps += 1
        
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = train_loss / args.log_interval
            avg_mse = train_mse / steps
            avg_mae = train_mae / steps
            avg_rmse = train_rmse / steps
            elapsed = time.time() - start_time
            
            avg_mse_denorm = denormalize_error(avg_mse, np.mean(norm_params['output_std']))
            avg_mae_denorm = denormalize_mae(avg_mae, np.mean(norm_params['output_std']))
            avg_rmse_denorm = denormalize_rmse(avg_rmse, np.mean(norm_params['output_std']))
            
            log_msg = f'Train Epoch: {ep} [{batch_idx * batch_size}/{len(train_dataset)} ({100. * batch_idx / (len(train_dataset) // batch_size):.0f}%)] | Loss: {avg_loss:.6f} (norm) | MSE: {avg_mse_denorm:.2f}°\u00b2 | MAE: {avg_mae_denorm:.2f}° | RMSE: {avg_rmse_denorm:.2f}° | Time: {elapsed:.1f}s'
            print(log_msg)
            
            train_loss = 0
            train_mse = 0
            train_mae = 0
            train_rmse = 0
            steps = 0
            start_time = time.time()
    
    epoch_loss = train_loss / max(steps, 1)
    return epoch_loss


def validate():
    global best_test_loss, best_mse_denorm, best_rmse_denorm
    model.eval()
    test_loss = 0
    test_mse = 0
    test_mae = 0
    test_rmse = 0
    n_batches = 0
    
    with torch.no_grad():
        for data, target in data_generator(test_dataset, batch_size, shuffle=False):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            mse, mae, rmse = calculate_metrics(output, target)
            test_mse += mse
            test_mae += mae
            test_rmse += rmse
            n_batches += 1
    
    test_loss /= len(test_dataset)
    test_mse /= n_batches
    test_mae /= n_batches
    test_rmse /= n_batches
    
    test_mse_denorm = denormalize_error(test_mse, np.mean(norm_params['output_std']))
    test_mae_denorm = denormalize_mae(test_mae, np.mean(norm_params['output_std']))
    test_rmse_denorm = denormalize_rmse(test_rmse, np.mean(norm_params['output_std']))
    
    log_msg = f'\nValidation | MSE: {test_mse_denorm:.2f}°\u00b2 | MAE: {test_mae_denorm:.2f}\u00b0 | RMSE: {test_rmse_denorm:.2f}\u00b0 (degrees)\n'
    print(log_msg)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_mse_denorm = denormalize_error(test_mse, np.mean(norm_params['output_std']))
        best_rmse_denorm = np.sqrt(best_mse_denorm)
        
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'mse_denorm': best_mse_denorm,
            'rmse_denorm': best_rmse_denorm,
            'config': {
                'input_size': input_size,
                'output_size': output_size,
                'channel_sizes': channel_sizes,
                'kernel_size': kernel_size,
                'dropout': args.dropout
            }
        }, args.save_path)
        
        print(f'Best model saved to {args.save_path} (MSE: {best_mse_denorm:.2f}°\u00b2, RMSE: {best_rmse_denorm:.2f}°)')

    return test_loss


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting Training - IMU to Joint Angle Prediction")
    print("=" * 80 + "\n")
    
    training_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss = train(epoch)
        val_loss = validate()
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        epoch_msg = f'Epoch {epoch}/{epochs} completed in {epoch_time:.1f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}'
        print(epoch_msg)
        
        if epoch % 10 == 0:
            lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f'Learning rate updated to {lr}')
    
    total_time = time.time() - training_start
    
    best_rmse_denorm = np.sqrt(best_mse_denorm)
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Total training time: {total_time:.1f}s")
    print(f"Best validation MSE: {best_mse_denorm:.2f}°\u00b2 (RMSE: {best_rmse_denorm:.2f}°)")
    print(f"Model saved to: {args.save_path}")
    print(f"Normalization saved to: {args.norm_path}")
    print(f"Log file: {log_file}")
    print("=" * 80)