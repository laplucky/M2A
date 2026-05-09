# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' 

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DEFAULT_BASE="./Qwen3-8B-Base"
DEFAULT_AGENT="./Agent-8B"
DEFAULT_REASON="./Reasoning-8B"

DEFAULT_OUTPUT="./output/M2A-Agent-8B"

DEFAULT_DATA="./data/scale_calibration.json"

BASE_MODEL=${1:-$DEFAULT_BASE}
AGENT_MODEL=${2:-$DEFAULT_AGENT}
REASON_MODEL=${3:-$DEFAULT_REASON}
DATA_FILE=${4:-$DEFAULT_DATA}
OUTPUT_DIR=${5:-$DEFAULT_OUTPUT}


RADIUS=${RADIUS:-5}
# Configurable parameters (via environment variables)
MAX_SAMPLES=${MAX_SAMPLES:-100}
LAYERS_START=${LAYERS_START:-0}
LAYERS_END=${LAYERS_END:-}  # Set to null to use all layers
HEADS=${HEADS:-"all"}
MERGE_TYPES=${MERGE_TYPES:-"qkvof"}
COMPUTE_PRECISION=${COMPUTE_PRECISION:-"fp32"}
LAMBDA_RIDGE=${LAMBDA_RIDGE:-1e-4}
CG_MAXIT=${CG_MAXIT:-100}
CG_TOL=${CG_TOL:-1e-5}

# specific parameters
BETA=${BETA:-1.1}              # Scaling factor for dynamic alpha
K_THRESHOLD=${K_THRESHOLD:-0}  # Threshold multiplier for similarity mask
WINDOW_SIZE=${WINDOW_SIZE:-3}   # Window size for similarity smoothing

# Device configuration
QK_DEVICE=${QK_DEVICE:-"auto"}
VO_DEVICE=${VO_DEVICE:-"auto"}
FFN_DEVICE=${FFN_DEVICE:-"auto"}

# Constraint parameter configuration
Q_ROWS=${Q_ROWS:-8}
K_ROWS=${K_ROWS:-8}
V_ROWS=${V_ROWS:-4}
O_ROWS=${O_ROWS:-4}
FFN_ROWS=${FFN_ROWS:-4}
W_Q=${W_Q:-1.0}
W_K=${W_K:-1.0}
W_V=${W_V:-1.0}
W_O=${W_O:-1.0}
W_FFN=${W_FFN:-1.0}
READOUT_DIRS=${READOUT_DIRS:-2}

# Sequence length limit
MAX_SEQ_LEN=${MAX_SEQ_LEN:-15000}

# Model name
MODEL_NAME=${MODEL_NAME:-"m2a"}

function show_help() {
    echo -e "${GREEN}M2A: Adaptive Similarity-Aware Projection Merge${NC}"
    echo ""
    echo "Usage: $0 [base_model] [agent_model] [reason_model] [data_file] [output_dir]"
    echo ""
    echo -e "${YELLOW}Positional Arguments:${NC}"
    echo "  base_model     Base model path (default: $DEFAULT_BASE)"
    echo "  agent_model    Agent model path (default: $DEFAULT_AGENT)"
    echo "  reason_model   Reasoning model path (default: $DEFAULT_REASON)"
    echo "  data_file      Training data file (default: $DEFAULT_DATA)"
    echo "  output_dir     Output directory (default: $DEFAULT_OUTPUT)"
    echo ""
    echo -e "${YELLOW}Environment Variable Configuration:${NC}"
    echo ""
    echo -e "${BLUE}Basic Parameters:${NC}"
    echo "  MAX_SAMPLES        Maximum number of samples (default: 200)"
    echo "  LAYERS_START       Start layer index, inclusive (default: 0)"
    echo "  LAYERS_END         End layer index, exclusive (default: 36)"
    echo "  HEADS              Attention heads to process (default: all)"
    echo "  MERGE_TYPES        Merge types (default: qkvof)"
    echo "  COMPUTE_PRECISION  Compute precision (default: fp32)"
    echo ""
    echo -e "${BLUE}M2A Specific:${NC}"
    echo "  BETA               Scaling factor for dynamic alpha (default: 1.0)"
    echo "                     α_l = BETA × ||ΔW_agent||_F / ||ΔW_reason||_F"
    echo "  K_THRESHOLD        Threshold multiplier for similarity mask (default: 0.5)"
    echo "                     M_l = 1 if S̃_l ≥ μ - K_THRESHOLD × σ"
    echo "  WINDOW_SIZE        Window size for similarity smoothing (default: 3)"
    echo ""
    echo -e "${BLUE}Solver Parameters:${NC}"
    echo "  LAMBDA_RIDGE       Ridge regression parameter (default: 1e-4)"
    echo "  CG_MAXIT           CG maximum iterations (default: 100)"
    echo "  CG_TOL             CG convergence tolerance (default: 1e-5)"
    echo ""
    echo -e "${BLUE}Device Configuration:${NC}"
    echo "  QK_DEVICE          QK compute device (default: auto)"
    echo "  VO_DEVICE          VO compute device (default: auto)"
    echo "  FFN_DEVICE         FFN compute device (default: auto)"
    echo ""
    echo -e "${BLUE}Constraint Parameters:${NC}"
    echo "  Q_ROWS, K_ROWS, V_ROWS, O_ROWS, FFN_ROWS"
    echo "  W_Q, W_K, W_V, W_O, W_FFN"
    echo "  READOUT_DIRS       Number of output readout directions (default: 2)"
    echo ""
    echo -e "${BLUE}Sequence Length Control:${NC}"
    echo "  MAX_SEQ_LEN        Maximum sequence length limit (default: 12800)"
    echo ""
    echo -e "${YELLOW}Key Features:${NC}"
    echo "  ✨ Automatic merge coefficient calculation per layer"
    echo "  ✨ Automatic layer selection based on similarity"
    echo "  ✨ Format conflict detection and protection"
    echo "  ✨ Null-space projection for capability preservation"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  # Basic usage with default parameters"
    echo "  $0"
    echo ""
    echo "  # Custom layer range"
    echo "  LAYERS_START=10 LAYERS_END=30 $0"
    echo ""
    echo "  # Adjust M2A sensitivity"
    echo "  BETA=1.5 K_THRESHOLD=0.3 $0"
    echo ""
    echo "  # High precision computation"
    echo "  COMPUTE_PRECISION=fp64 LAMBDA_RIDGE=1e-5 $0"
    echo ""
    echo "  # Multi-GPU configuration"
    echo "  QK_DEVICE=cuda:0 VO_DEVICE=cuda:1 FFN_DEVICE=cuda:2 $0"
}

# Check for help argument
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

echo -e "${BLUE}=======================================================================${NC}"
echo -e "${BLUE}        M2A: Adaptive Similarity-Aware Projection Merge${NC}"
echo -e "${BLUE}=======================================================================${NC}"
echo -e "${GREEN}📁 Base Model: ${NC}$BASE_MODEL"
echo -e "${GREEN}📁 Agent Model: ${NC}$AGENT_MODEL"
echo -e "${GREEN}📁 Reasoning Model: ${NC}$REASON_MODEL"
echo -e "${GREEN}📁 Training Data: ${NC}$DATA_FILE"
echo -e "${GREEN}📁 Output Directory: ${NC}$OUTPUT_DIR"
echo ""
echo -e "${YELLOW}Configuration Parameters:${NC}"
echo "  Max samples: $MAX_SAMPLES"
echo "  Layers: [$LAYERS_START, $LAYERS_END)"
echo "  Heads: $HEADS"
echo "  Merge types: $MERGE_TYPES"
echo "  Compute precision: $COMPUTE_PRECISION"
echo ""
echo -e "${YELLOW}M2A Parameters:${NC}"
echo "  Beta (α scaling): $BETA"
echo "  K threshold: $K_THRESHOLD"
echo "  Window size: $WINDOW_SIZE"
echo ""
echo -e "${YELLOW}Device Configuration:${NC}"
echo "  QK=$QK_DEVICE, VO=$VO_DEVICE, FFN=$FFN_DEVICE"
echo "  Sequence length limit: $MAX_SEQ_LEN tokens"
echo -e "${BLUE}=======================================================================${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "\n${BLUE}🔄 Starting M2A...${NC}"

# Record start time
START_TIME=$(date +%s)

# Build layer selection arguments
LAYER_ARGS="--layers_start $LAYERS_START"
if [[ -n "$LAYERS_END" ]]; then
    LAYER_ARGS="$LAYER_ARGS --layers_end $LAYERS_END"
fi

# Execute M2A
python m2a_main.py \
    --base "$BASE_MODEL" \
    --agent "$AGENT_MODEL" \
    --reason "$REASON_MODEL" \
    --texts_r "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --max_samples_r $MAX_SAMPLES \
    $LAYER_ARGS \
    --heads "$HEADS" \
    --merge_types "$MERGE_TYPES" \
    --compute_precision "$COMPUTE_PRECISION" \
    --lambda_ridge $LAMBDA_RIDGE \
    --cg_maxit $CG_MAXIT \
    --cg_tol $CG_TOL \
    --beta $BETA \
    --k_threshold $K_THRESHOLD \
    --window_size $WINDOW_SIZE \
    --q_rows_per_text $Q_ROWS \
    --k_rows_per_text $K_ROWS \
    --v_rows_per_text $V_ROWS \
    --o_rows_per_text $O_ROWS \
    --ffn_rows_per_text $FFN_ROWS \
    --w_q $W_Q \
    --w_k $W_K \
    --w_v $W_V \
    --w_o $W_O \
    --w_ffn $W_FFN \
    --readout_dirs $READOUT_DIRS \
    --qk_device "$QK_DEVICE" \
    --vo_device "$VO_DEVICE" \
    --ffn_device "$FFN_DEVICE" \
    --max_seq_len $MAX_SEQ_LEN \
    --use_hooks \
    --neigh_radius $RADIUS \
    --seed 40


EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}=======================================================================${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✅ M2A completed successfully! Elapsed: ${DURATION}s${NC}"
    echo -e "${GREEN}📁 Output directory: $OUTPUT_DIR${NC}"
    echo ""

    # Check merged model
    MERGED_MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
    if [[ -d "$MERGED_MODEL_DIR" ]]; then
        echo -e "${GREEN}🤖 Merged model: $MERGED_MODEL_DIR${NC}"
        echo ""
        echo -e "${YELLOW}📊 Model files:${NC}"
        ls -lh "$MERGED_MODEL_DIR" | head -20
        echo ""
    fi

    # Check stats file
    STATS_FILE="$OUTPUT_DIR/asap_merge_stats.json"
    if [[ -f "$STATS_FILE" ]]; then
        echo -e "${YELLOW}📊 Merge statistics:${NC}"
        python3 -c "
import json
try:
    with open('$STATS_FILE', 'r') as f:
        stats = json.load(f)

    merge_stats = stats.get('merge_stats', {})
    print('  Layers merged:', merge_stats.get('layers_merged', 'N/A'))
    print('  Layers protected:', merge_stats.get('layers_protected', 'N/A'))
    print('  Total params modified:', f\"{merge_stats.get('total_params_modified', 0):,}\")

    # Show some layer details
    layer_details = merge_stats.get('layer_details', {})
    if layer_details:
        print('\\n  Sample Layer Decisions:')
        for layer_id in sorted([int(k) for k in layer_details.keys()])[:5]:
            detail = layer_details[str(layer_id)]
            action = detail.get('action', 'N/A')
            alpha = detail.get('alpha', 0)
            sim = detail.get('similarity', 0)
            mask = detail.get('mask', 0)
            print(f'    Layer {layer_id}: {action} (α={alpha:.4f}, S={sim:.4f}, M={mask})')
except Exception as e:
    print('  Failed to read statistics:', e)
"
        echo ""
        echo -e "${GREEN}📄 Full statistics: $STATS_FILE${NC}"
    fi

    echo ""
    echo -e "${YELLOW}🎉 M2A pipeline complete!${NC}"
    echo -e "${GREEN}📄 Using the merged model:${NC}"
    echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
    echo "  model = AutoModelForCausalLM.from_pretrained('$MERGED_MODEL_DIR')"
    echo "  tokenizer = AutoTokenizer.from_pretrained('$MERGED_MODEL_DIR')"

else
    echo -e "${RED}❌ M2A failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}Please check the error messages and retry${NC}"
fi

echo -e "${BLUE}=======================================================================${NC}"

exit $EXIT_CODE
