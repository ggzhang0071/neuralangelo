# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# {experiment_name} {path_to_video} {skip_frame_rate} {scene_type}
base_name=$(basename ${2})
image_path=datasets/${base_name%.*}_skip${3}
bash projects/neuralangelo/scripts/run_ffmpeg.sh ${2} ${3}
bash projects/neuralangelo/scripts/run_colmap.sh ${image_path}
python3   projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${image_path}/dense --scene_type ${4}
python3 projects/neuralangelo/scripts/generate_config.py --experiment_name ${1} --data_dir ${image_path}/dense --scene_type ${4} --auto_exposure_wb


PATH_TO_VIDEO=./examples/lego.mp4
SKIP_FRAME_RATE=2  # Set this to a larger value (e.g. 24) for small video motions and smaller value (e.g.) for large video motions.
bash projects/neuralangelo/scripts/run_ffmpeg.sh ${PATH_TO_VIDEO} ${SKIP_FRAME_RATE}

PATH_TO_IMAGES=./datasets/lego_skip2
bash projects/neuralangelo/scripts/run_colmap.sh ${PATH_TO_IMAGES}