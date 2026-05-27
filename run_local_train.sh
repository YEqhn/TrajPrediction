#!/bin/bash

set -e

echo "=========================================="
echo "TCN Local Training Runner"
echo "=========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TRAIN_SCRIPT="${PROJECT_ROOT}/TCN/prediction_hip_test/prediction_hip_test.py"

mkdir -p "${LOG_DIR}"

log_file="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
config_file="${LOG_DIR}/config_$(date +%Y%m%d_%H%M%S).txt"

exec > >(tee -a "${log_file}") 2>&1

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Log file: ${log_file}"
echo ""

echo "=========================================="
echo "1. Conda Environment Activation"
echo "=========================================="
echo ""

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    echo "[INFO] Miniconda found, sourcing conda.sh..."
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    echo "[INFO] Anaconda found, sourcing conda.sh..."
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    echo "[INFO] Conda found at /opt/conda, sourcing conda.sh..."
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "[ERROR] Conda installation not found!"
    echo "[INFO] Please install Miniconda or Anaconda"
    exit 1
fi

if conda env list | grep -q "^TCN "; then
    echo "[INFO] TCN conda environment found"
    echo "[INFO] Activating TCN environment..."
    conda activate TCN
    
    echo "[INFO] Python location: $(which python)"
    echo "[INFO] Python version: $(python --version)"
else
    echo "[WARN] TCN conda environment not found"
    echo "[INFO] Creating TCN environment..."
    
    conda create -n TCN python=3.9 -y
    conda activate TCN
    
    echo "[INFO] Installing dependencies in TCN environment..."
    conda install -n TCN -c pytorch pytorch torchvision torchaudio cudatoolkit=11.8 -y
    pip install numpy pandas scipy matplotlib
fi

echo ""
echo "[INFO] TCN environment ready"
echo ""

echo "=========================================="
echo "2. Environment Check"
echo "=========================================="
echo ""

echo "[CHECK] Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "       Python version: ${python_version}"

echo ""
echo "[CHECK] CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "       NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "       (nvidia-smi query failed)"
else
    echo "       [WARN] No NVIDIA GPU detected or nvidia-smi not available"
fi

echo ""
python -c "import torch; print('       PyTorch version:', torch.__version__); print('       CUDA available:', torch.cuda.is_available()); print('       CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "       [WARN] PyTorch not installed"

echo ""
echo "[CHECK] Required packages..."
for pkg in numpy pandas scipy; do
    if python -c "import ${pkg}" 2>/dev/null; then
        version=$(python -c "import ${pkg}; print(${pkg}.__version__)")
        echo "       ${pkg}: ${version} [OK]"
    else
        echo "       ${pkg}: [NOT INSTALLED]"
    fi
done

echo ""
echo "=========================================="
echo "3. Training Parameters Configuration"
echo "=========================================="
echo ""

BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-0.001}
HIDDEN_UNITS=${HIDDEN_UNITS:-64}
KERNEL_SIZE=${KERNEL_SIZE:-5}
LEVELS=${LEVELS:-6}
DROPOUT=${DROPOUT:-0.2}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
OPTIMIZER=${OPTIMIZER:-Adam}
RANDOM_SEED=${RANDOM_SEED:-1111}
LOG_INTERVAL=${LOG_INTERVAL:-50}
DATA_PATH=${DATA_PATH:-/home/yeqhn/Desktop/TCN/data_processed}
WINDOW_SECONDS=${WINDOW_SECONDS:-1.0}
PREDICT_MS=${PREDICT_MS:-100}
SAMPLE_RATE=${SAMPLE_RATE:-200}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
USE_CUDA=${USE_CUDA:-auto}

cat > "${config_file}" << EOF
==========================================
Training Configuration
==========================================
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Environment: TCN conda environment
Batch Size: ${BATCH_SIZE}
Epochs: ${EPOCHS}
Learning Rate: ${LEARNING_RATE}
Hidden Units: ${HIDDEN_UNITS}
Kernel Size: ${KERNEL_SIZE}
Levels: ${LEVELS}
Dropout: ${DROPOUT}
Gradient Clip: ${GRADIENT_CLIP}
Optimizer: ${OPTIMIZER}
Random Seed: ${RANDOM_SEED}
Log Interval: ${LOG_INTERVAL}
Data Path: ${DATA_PATH}
Window Duration: ${WINDOW_SECONDS}s
Prediction Horizon: ${PREDICT_MS}ms
Sample Rate: ${SAMPLE_RATE}Hz
Train Ratio: ${TRAIN_RATIO}
CUDA: ${USE_CUDA}
==========================================
EOF

echo "[INFO] Configuration saved to: ${config_file}"
cat "${config_file}"

echo ""
echo "=========================================="
echo "4. Pre-training Validation"
echo "=========================================="
echo ""

echo "[CHECK] Training script exists: ${TRAIN_SCRIPT}"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "[ERROR] Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

echo "[CHECK] Data path exists: ${DATA_PATH}"
if [ ! -d "${DATA_PATH}" ]; then
    echo "[WARN] Data path does not exist: ${DATA_PATH}"
    echo "[INFO] Training will likely fail if data is required"
fi

echo "[CHECK] Project structure..."
if [ -d "${PROJECT_ROOT}/TCN" ]; then
    echo "       TCN module: [OK]"
fi
if [ -f "${PROJECT_ROOT}/TCN/tcn.py" ]; then
    echo "       TCN core: [OK]"
fi
if [ -f "${PROJECT_ROOT}/TCN/prediction_hip_test/utils.py" ]; then
    echo "       Utils: [OK]"
fi
if [ -f "${PROJECT_ROOT}/TCN/prediction_hip_test/model.py" ]; then
    echo "       Model: [OK]"
fi

echo ""
echo "=========================================="
echo "5. Starting Training"
echo "=========================================="
echo ""

training_start_time=$(date +%s)

CUDA_FLAG=""
if [ "${USE_CUDA}" = "auto" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_FLAG="--cuda"
        echo "[INFO] Auto-detected CUDA, enabling GPU acceleration"
    else
        echo "[INFO] CUDA not available, running on CPU"
    fi
elif [ "${USE_CUDA}" = "true" ] || [ "${USE_CUDA}" = "1" ]; then
    CUDA_FLAG="--cuda"
fi

echo "[INFO] Executing training script..."
echo "[INFO] Command: python ${TRAIN_SCRIPT} \\"
echo "       --batch_size ${BATCH_SIZE} \\"
echo "       --epochs ${EPOCHS} \\"
echo "       --lr ${LEARNING_RATE} \\"
echo "       --nhid ${HIDDEN_UNITS} \\"
echo "       --ksize ${KERNEL_SIZE} \\"
echo "       --levels ${LEVELS} \\"
echo "       --dropout ${DROPOUT} \\"
echo "       --clip ${GRADIENT_CLIP} \\"
echo "       --optim ${OPTIMIZER} \\"
echo "       --seed ${RANDOM_SEED} \\"
echo "       --log_interval ${LOG_INTERVAL} \\"
echo "       --data_path ${DATA_PATH} \\"
echo "       --window_seconds ${WINDOW_SECONDS} \\"
echo "       --predict_ms ${PREDICT_MS} \\"
echo "       --sample_rate ${SAMPLE_RATE} \\"
echo "       --train_ratio ${TRAIN_RATIO} ${CUDA_FLAG}"
echo ""

python "${TRAIN_SCRIPT}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --nhid ${HIDDEN_UNITS} \
    --ksize ${KERNEL_SIZE} \
    --levels ${LEVELS} \
    --dropout ${DROPOUT} \
    --clip ${GRADIENT_CLIP} \
    --optim ${OPTIMIZER} \
    --seed ${RANDOM_SEED} \
    --log_interval ${LOG_INTERVAL} \
    --data_path ${DATA_PATH} \
    --window_seconds ${WINDOW_SECONDS} \
    --predict_ms ${PREDICT_MS} \
    --sample_rate ${SAMPLE_RATE} \
    --train_ratio ${TRAIN_RATIO} \
    ${CUDA_FLAG}

training_exit_code=$?
training_end_time=$(date +%s)
training_duration=$((training_end_time - training_start_time))

echo ""
echo "=========================================="
echo "6. Training Completion"
echo "=========================================="
echo ""

if [ $training_exit_code -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully!"
else
    echo "[ERROR] Training failed with exit code: ${training_exit_code}"
fi

echo "[INFO] Training duration: ${training_duration} seconds ($(($training_duration / 60)) minutes)"
echo "[INFO] Full log saved to: ${log_file}"
echo "[INFO] Configuration saved to: ${config_file}"

echo ""
echo "=========================================="
echo "Log file content (last 50 lines):"
echo "=========================================="
tail -n 50 "${log_file}"

exit ${training_exit_code}