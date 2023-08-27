EXPERIMENT=chair
PATH_TO_VIDEO=chair.mp4  
SKIP_FRAME_RATE=2  # Set this to a larger value (e.g. 24) for small video motions and smaller value (e.g.) for large video motions.
SCENE_TYPE=object  # {outdoor,indoor,object}
bash projects/neuralangelo/scripts/preprocess.sh ${EXPERIMENT} ${PATH_TO_VIDEO} ${SKIP_FRAME_RATE} ${SCENE_TYPE}

timestamp=$(date +%Y%m%d%H%M%S)

GROUP=object
NAME=chair
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun   --nproc_per_node=${GPUS}  train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar \
    --resume \
    --wandb \
    --wandb_name =${NAME} \
    --checkpoint=logs/${GROUP}/${NAME}/epoch_01600_iteration_000080000_checkpoint.pt  \
     2>&1 |tee Logs/$timestamp.log


CHECKPOINT=logs/${GROUP}/${NAME}/epoch_00400_iteration_000020000_checkpoint.pt
OUTPUT_MESH=lego.ply
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
RESOLUTION=2048
BLOCK_RES=128
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES} \
    --textured
