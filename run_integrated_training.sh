#!/bin/bash

set -e

echo "=========================================="
echo "Integrated TCN + TimesNet + NormalTrajMemory"
echo "End-to-End Training Script"
echo "=========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
MODELS_DIR="${PROJECT_ROOT}/models"

TRAIN_SCRIPT="${PROJECT_ROOT}/integrated_training.py"
ANOMALY_TEST_SCRIPT="${PROJECT_ROOT}/anomaly_test.py"

mkdir -p "${LOG_DIR}"
mkdir -p "${MODELS_DIR}"

log_file="${LOG_DIR}/integrated_training_$(date +%Y%m%d_%H%M%S).log"
config_file="${LOG_DIR}/config_$(date +%Y%m%d_%H%M%S).txt"

exec > >(tee -a "${log_file}") 2>&1

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Log file: ${log_file}"
echo ""

echo "=========================================="
echo "1. Conda Environment Setup"
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
elif conda env list | grep -q "^base "; then
    echo "[INFO] Base conda environment found"
    echo "[INFO] Using base environment..."
else
    echo "[WARN] TCN conda environment not found"
    echo "[INFO] Please create TCN environment first"
fi

echo ""
echo "[INFO] Python location: $(which python)"
echo "[INFO] Python version: $(python --version)"
echo ""

echo "=========================================="
echo "2. TimesNet Dependencies Installation"
echo "=========================================="
echo ""

echo "[CHECK] Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print('       PyTorch:', torch.__version__); print('       CUDA available:', torch.cuda.is_available())"
else
    echo "[WARN] PyTorch not installed. Installing..."
    conda install -c pytorch pytorch torchvision -y 2>/dev/null || pip install torch torchvision
fi

echo ""
echo "[CHECK] Checking required packages..."
for pkg in numpy pandas scipy scikit-learn; do
    if python -c "import ${pkg}" 2>/dev/null; then
        version=$(python -c "import ${pkg}; print(${pkg}.__version__)")
        echo "       ${pkg}: ${version} [OK]"
    else
        echo "       ${pkg}: [NOT INSTALLED] - Will install"
        pip install ${pkg}
    fi
done

echo ""
echo "[CHECK] Checking sktime..."
if python -c "import sktime" 2>/dev/null; then
    echo "       sktime: [OK]"
else
    echo "       sktime: [NOT INSTALLED] - Installing..."
    pip install sktime statsmodels
fi

echo ""
echo "=========================================="
echo "3. Environment Validation"
echo "=========================================="
echo ""

echo "[CHECK] CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "       NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "       (nvidia-smi query failed)"
else
    echo "       [WARN] No NVIDIA GPU detected or nvidia-smi not available"
fi

echo ""
echo "[CHECK] Python environment..."
python -c "import torch; print('       PyTorch:', torch.__version__); print('       CUDA:', torch.cuda.is_available())" 2>/dev/null || echo "       [ERROR] PyTorch import failed"
python -c "import numpy; print('       NumPy:', numpy.__version__)" 2>/dev/null || echo "       [ERROR] NumPy import failed"
python -c "import pandas; print('       Pandas:', pandas.__version__)" 2>/dev/null || echo "       [ERROR] Pandas import failed"
python -c "import sklearn; print('       Scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "       [ERROR] Scikit-learn import failed"

echo ""
echo "=========================================="
echo "4. Training Parameters Configuration"
echo "=========================================="
echo ""

BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-50}
TIMESNET_EPOCHS=${TIMESNET_EPOCHS:-20}
LEARNING_RATE=${LEARNING_RATE:-0.001}
HIDDEN_UNITS=${HIDDEN_UNITS:-64}
KERNEL_SIZE=${KERNEL_SIZE:-5}
LEVELS=${LEVELS:-6}
DROPOUT=${DROPOUT:-0.2}
DATA_PATH=${DATA_PATH:-/home/yeqhn/Desktop/TCN/data_processed}
WINDOW_SECONDS=${WINDOW_SECONDS:-1.0}
PREDICT_MS=${PREDICT_MS:-100}
SAMPLE_RATE=${SAMPLE_RATE:-200}
ANOMALY_THRESHOLD=${ANOMALY_THRESHOLD:-0.5}
CORRECTION_WEIGHT=${CORRECTION_WEIGHT:-0.7}
MEMORY_SIZE=${MEMORY_SIZE:-100}
TCN_SAVE_PATH=${TCN_SAVE_PATH:-/home/yeqhn/Desktop/TCN/models/tcn_model.pt}
TIMESNET_SAVE_PATH=${TIMESNET_SAVE_PATH:-/home/yeqhn/Desktop/TCN/models/timesnet_anomaly_detector.pt}
RUN_TESTS=${RUN_TESTS:-true}
USE_CUDA=${USE_CUDA:-true}

cat > "${config_file}" << EOF
==========================================
Integrated Training Configuration
==========================================
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Environment: TCN conda environment
==========================================
General:
  Batch Size: ${BATCH_SIZE}
  Learning Rate: ${LEARNING_RATE}
  Use CUDA: ${USE_CUDA}
  Data Path: ${DATA_PATH}
  Window Duration: ${WINDOW_SECONDS}s
  Prediction Horizon: ${PREDICT_MS}ms
  Sample Rate: ${SAMPLE_RATE}Hz

TCN Module:
  Epochs: ${EPOCHS}
  Hidden Units: ${HIDDEN_UNITS}
  Kernel Size: ${KERNEL_SIZE}
  Levels: ${LEVELS}
  Dropout: ${DROPOUT}

TimesNet Module:
  Epochs: ${TIMESNET_EPOCHS}
  Anomaly Threshold: ${ANOMALY_THRESHOLD}

NormalTrajMemory Module:
  Memory Size: ${MEMORY_SIZE}
  Correction Weight: ${CORRECTION_WEIGHT}

Output Paths:
  TCN Model: ${TCN_SAVE_PATH}
  TimesNet Model: ${TIMESNET_SAVE_PATH}

Options:
  Run Tests: ${RUN_TESTS}
==========================================
EOF

echo "[INFO] Configuration saved to: ${config_file}"
cat "${config_file}"

echo ""
echo "=========================================="
echo "5. Pre-training Validation"
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

echo "[CHECK] Project modules..."
if [ -f "${PROJECT_ROOT}/TCN/tcn.py" ]; then
    echo "       TCN core: [OK]"
fi
if [ -f "${PROJECT_ROOT}/TCN/normal_traj_memory.py" ]; then
    echo "       NormalTrajMemory: [OK]"
fi
if [ -f "${PROJECT_ROOT}/TimesNet_Model/timesnet_anomaly_detector.py" ]; then
    echo "       TimesNet Anomaly Detector: [OK]"
fi
if [ -f "${PROJECT_ROOT}/integrated_training.py" ]; then
    echo "       Integrated Training: [OK]"
fi

echo ""
echo "=========================================="
echo "6. Starting Integrated Training"
echo "=========================================="
echo ""

training_start_time=$(date +%s)

CUDA_FLAG=""
if [ "${USE_CUDA}" = "true" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_FLAG="--cuda"
        echo "[INFO] CUDA detected, enabling GPU acceleration"
    else
        echo "[INFO] CUDA not available, running on CPU"
    fi
else
    echo "[INFO] CUDA disabled by configuration"
fi

TEST_FLAG=""
if [ "${RUN_TESTS}" = "true" ]; then
    TEST_FLAG="--run_tests"
    echo "[INFO] Anomaly tests will be run after training"
else
    echo "[INFO] Anomaly tests disabled"
fi

echo ""
echo "[COMMAND] python ${TRAIN_SCRIPT} \\"
echo "          --batch_size ${BATCH_SIZE} \\"
echo "          --tcn_epochs ${EPOCHS} \\"
echo "          --timesnet_epochs ${TIMESNET_EPOCHS} \\"
echo "          --lr ${LEARNING_RATE} \\"
echo "          --data_path ${DATA_PATH} \\"
echo "          --window_seconds ${WINDOW_SECONDS} \\"
echo "          --predict_ms ${PREDICT_MS} \\"
echo "          --sample_rate ${SAMPLE_RATE} \\"
echo "          --anomaly_threshold ${ANOMALY_THRESHOLD} \\"
echo "          --tcn_save_path ${TCN_SAVE_PATH} \\"
echo "          --timesnet_save_path ${TIMESNET_SAVE_PATH} \\"
echo "          --log_dir ${LOG_DIR} ${CUDA_FLAG} ${TEST_FLAG}"
echo ""

cd "${PROJECT_ROOT}"

python ${TRAIN_SCRIPT} \
    --batch_size ${BATCH_SIZE} \
    --tcn_epochs ${EPOCHS} \
    --timesnet_epochs ${TIMESNET_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --data_path ${DATA_PATH} \
    --window_seconds ${WINDOW_SECONDS} \
    --predict_ms ${PREDICT_MS} \
    --sample_rate ${SAMPLE_RATE} \
    --anomaly_threshold ${ANOMALY_THRESHOLD} \
    --tcn_save_path ${TCN_SAVE_PATH} \
    --timesnet_save_path ${TIMESNET_SAVE_PATH} \
    --log_dir ${LOG_DIR} \
    ${CUDA_FLAG} \
    ${TEST_FLAG}

training_exit_code=$?
training_end_time=$(date +%s)
training_duration=$((training_end_time - training_start_time))

echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo ""
echo "[INFO] Training duration: ${training_duration}s ($(($training_duration / 60))m $(($training_duration % 60))s)"
echo "[INFO] Exit code: ${training_exit_code}"

if [ ${training_exit_code} -eq 0 ]; then
    echo "[INFO] Training completed successfully!"
    
    echo ""
    echo "=========================================="
    echo "7. Saved Models"
    echo "=========================================="
    echo ""
    
    if [ -f "${TCN_SAVE_PATH}" ]; then
        echo "       TCN Model: ${TCN_SAVE_PATH}"
        ls -lh "${TCN_SAVE_PATH}" 2>/dev/null || echo "       (file info unavailable)"
    fi
    
    if [ -f "${TIMESNET_SAVE_PATH}" ]; then
        echo "       TimesNet Model: ${TIMESNET_SAVE_PATH}"
        ls -lh "${TIMESNET_SAVE_PATH}" 2>/dev/null || echo "       (file info unavailable)"
    fi
    
    echo ""
    echo "=========================================="
    echo "8. Logs"
    echo "=========================================="
    echo ""
    echo "       Training log: ${log_file}"
    echo "       Configuration: ${config_file}"
    
    if [ -d "${LOG_DIR}" ]; then
        echo ""
        echo "       Recent log files:"
        ls -lt "${LOG_DIR}"/*.log 2>/dev/null | head -5 || echo "       (no log files found)"
    fi
else
    echo "[ERROR] Training failed with exit code ${training_exit_code}"
    echo "[INFO] Check log file for details: ${log_file}"
fi

echo ""
echo "=========================================="
echo "Training Pipeline Complete"
echo "=========================================="

exit ${training_exit_code}