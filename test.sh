SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}

TEST_CT_MR_ROOT="/ViPSAM/ViPSAM_/data/test/image"
TEST_MASK_ROOT="/ViPSAM/ViPSAM_/data/test/Mask"

SAM_CKPT="/ViPSAM/work_dir/SAM/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH="/ViPSAM/work_dir/ViPSAM/ViPSAM_sam.pth"
IMAGE_SIZE=1024
PROMPT_TYPE="box"

RESULTS_DIR="/ViPSAM/ViPSAM_/output/result"
MODEL_NAME="test"

BATCH_SIZE=1
NUM_WORKERS=4
GPU_DEVICE=0

USE_LORA=true
LORA_RANK=8
LORA_ALPHA=8

python test.py \
    --test_ct_mr_root ${TEST_CT_MR_ROOT} \
    --test_mask_root ${TEST_MASK_ROOT} \
    --sam_ckpt ${SAM_CKPT} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --image_size ${IMAGE_SIZE} \
    --results_dir ${RESULTS_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --gpu_device ${GPU_DEVICE} \
    --use_lora ${USE_LORA} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --prompt_type ${PROMPT_TYPE} \
    --model_name ${MODEL_NAME} \
