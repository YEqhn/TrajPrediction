#!/bin/bash
set -e

PROJECT_ROOT="/home/yeqhn/Desktop/TCN"
LOG_DIR="${PROJECT_ROOT}/logs"

show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -l, --log FILE         Log file path (required)"
    echo "  -c, --command CMD      Training command to execute"
    echo "  -e, --epochs N         Number of epochs"
    echo "  -b, --batch-size N     Batch size"
    echo "  -d, --data-path PATH   Data path"
    echo "  --no-cuda              Run without CUDA"
    echo ""
    echo "Example:"
    echo "  $0 -l results/exp/run_gpu.log --epochs 50 --batch-size 32"
    echo ""
    exit 1
}

USE_CUDA="true"
EPOCHS=""
BATCH_SIZE=""
DATA_PATH=""
LOG_FILE=""
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        -c|--command)
            COMMAND="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --no-cuda)
            USE_CUDA="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

if [ -z "$LOG_FILE" ]; then
    echo "Error: Log file path is required"
    echo "Use -l or --log to specify the log file"
    exit 1
fi

mkdir -p "$(dirname "$PROJECT_ROOT/$LOG_FILE")"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "TCN GPU Training via LocalSSH"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
echo "Log file: $PROJECT_ROOT/$LOG_FILE"
echo "CUDA: $USE_CUDA"
echo "=========================================="

ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 localhost << 'OUTER_EOF' 2>/dev/null || {
    echo "[ERROR] Cannot connect to localhost via SSH"
    echo "[INFO] Checking if SSH is configured..."
    echo "[INFO] Attempting to start SSH service..."
    
    if command -v systemctl &> /dev/null; then
        sudo systemctl start sshd 2>/dev/null || sudo systemctl start ssh 2>/dev/null || true
    fi
    
    if command -v service &> /dev/null; then
        sudo service ssh start 2>/dev/null || true
    fi
    
    sleep 2
    
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes localhost << 'INNER_EOF'
        echo "[INFO] SSH connection established"
    INNER_EOF
}

ssh -o StrictHostKeyChecking=no localhost << SSH_EOF
set -e

cd $PROJECT_ROOT

echo "=== SSH Connection Established ==="
echo "Hostname: \$(hostname)"
echo "Date: \$(date)"
echo ""

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

echo "=== Conda Environment Setup ==="
if conda env list | grep -q "^TCN "; then
    conda activate TCN
    echo "Activated TCN conda environment"
else
    echo "[WARN] TCN environment not found, using base environment"
fi

echo ""
echo "=== GPU Check ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "GPU Available"
else
    echo "[WARN] nvidia-smi not found, GPU not available"
fi

echo ""
echo "=== Python Environment ==="
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo ""
echo "=== Required Packages ==="
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"

echo ""
echo "=== Project Structure ==="
ls -la TCN/
ls -la TCN/prediction_hip_test/

echo ""
echo "=== Data Path Check ==="
DATA_PATH_DEFAULT="/home/yeqhn/Desktop/TCN/data_processed"
if [ -d "\$DATA_PATH_DEFAULT/ab06_ramp_csv" ]; then
    echo "Data path: \$DATA_PATH_DEFAULT"
    ls \$DATA_PATH_DEFAULT/ab06_ramp_csv/
else
    echo "[WARN] Data path not found: \$DATA_PATH_DEFAULT"
fi

echo ""
echo "=========================================="
echo "Starting Training via Background Process"
echo "=========================================="

TRAIN_SCRIPT="TCN/prediction_hip_test/prediction_hip_test.py"

TRAIN_CMD="python \${TRAIN_SCRIPT}"
TRAIN_CMD="\$TRAIN_CMD --epochs ${EPOCHS:-50}"
TRAIN_CMD="\$TRAIN_CMD --batch_size ${BATCH_SIZE:-32}"
TRAIN_CMD="\$TRAIN_CMD --data_path ${DATA_PATH:-/home/yeqhn/Desktop/TCN/data_processed}"
TRAIN_CMD="\$TRAIN_CMD --window_seconds 1.0"
TRAIN_CMD="\$TRAIN_CMD --predict_ms 100"
TRAIN_CMD="\$TRAIN_CMD --sample_rate 200"
TRAIN_CMD="\$TRAIN_CMD --train_ratio 0.8"
TRAIN_CMD="\$TRAIN_CMD --log_dir logs"

if [ "$USE_CUDA" = "true" ]; then
    TRAIN_CMD="\$TRAIN_CMD --cuda"
fi

echo "Training command: \$TRAIN_CMD"
echo ""

nohup bash -c "\$TRAIN_CMD" > $PROJECT_ROOT/$LOG_FILE 2>&1 &
TRAIN_PID=\$!

echo "Training started with PID: \$TRAIN_PID"
echo "Log file: $PROJECT_ROOT/$LOG_FILE"
echo ""

sleep 5

if ps -p \$TRAIN_PID > /dev/null; then
    echo "=== Training Process Running ==="
    echo "PID: \$TRAIN_PID"
    
    echo ""
    echo "=== Initial Log Output (first 30 lines) ==="
    head -n 30 $PROJECT_ROOT/$LOG_FILE || echo "No output yet"
    
    echo ""
    echo "=== GPU Status During Training ==="
    nvidia-smi || echo "GPU monitoring not available"
else
    echo "[ERROR] Training process terminated immediately"
    echo ""
    echo "=== Log File Content ==="
    cat $PROJECT_ROOT/$LOG_FILE
fi

echo ""
echo "=== Training is running in background ==="
echo "=== Monitor with: tail -f $PROJECT_ROOT/$LOG_FILE ==="
echo "=== Or check GPU with: nvidia-smi ==="
echo ""
echo "Connection established successfully!"
echo "Training PID: \$TRAIN_PID"

SSH_EOF

echo ""
echo "=========================================="
echo "Training initiated via LocalSSH"
echo "Log file: $PROJECT_ROOT/$LOG_FILE"
echo "=========================================="