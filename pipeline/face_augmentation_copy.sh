# $1은 파일명
IMG_NUMBER=$1
echo -e  'face_alignment 중....\n'

mkdir face_alignmented

cd /home/daehyeon/Face_Recognition
python Face_Alignment.py \
--img_path=/database/daehyeon/High_Resolution/frontal/${IMG_NUMBER}.png
echo -e '\n face_keypoints 추출 중...\n'

python dlib_keypoints.py

echo -e '\n 3d_keypoints 생성 중...\n'

export DIR_3DFAW=/home/daehyeon/DepthNets/depthnet-pytorch/3dfaw

NAME=exp1_lamb1_sd5_wgan_dnorm0.1_sigma0_fixm
cd /home/daehyeon/DepthNets/pipeline
OUT_FOLDER=dummy
FACE_FOLDER=/home/daehyeon/DepthNets/pipeline/face_alignmented

mkdir ${OUT_FOLDER}

python /home/daehyeon/DepthNets/depthnet-pytorch/export_to_facewarper_single.py \
--network=/home/daehyeon/DepthNets/depthnet-pytorch/architectures/depthnet_shallowd5.py \
--checkpoint=/home/daehyeon/DepthNets/depthnet-pytorch/results/${NAME}/models/100.pkl \
--src_kpts_file=${FACE_FOLDER}/keypoints_text/source.png.txt \
--tgt_kpts_file=${FACE_FOLDER}/keypoints_text/source.png.txt \
--src_img_file=${FACE_FOLDER}/source.png \
--tgt_img_file=${FACE_FOLDER}/source.png \
--output_dir=${FACE_FOLDER}/${OUT_FOLDER}/ \
--save_depth_path=${FACE_FOLDER}/keypoints_text \
--kpt_file_separator=' '

echo -e '\n 3d keypoints 만드는중.. \n'

python /home/daehyeon/DepthNets/pipeline/2d_to_3d.py

OUTPUT=rotating_data

echo -e '\n rotating_data 만드는중...  \n'

python /home/daehyeon/DepthNets/depthnet-pytorch/export_anim_to_facewarper.py \
--checkpoint=/home/daehyeon/DepthNets/depthnet-pytorch/results/${NAME}/models/100.pkl \
--network=/home/daehyeon/DepthNets/depthnet-pytorch/architectures/depthnet_shallowd5.py \
--axis=y \
--src_kpts_file=${FACE_FOLDER}/keypoints_text/3d_keypoints_txt \
--tgt_kpts_file=${FACE_FOLDER}/keypoints_text/3d_keypoints_txt \
--src_img_file=${FACE_FOLDER}/source.png \
--tgt_img_file=${FACE_FOLDER}/source.png \
--output_dir=${FACE_FOLDER}/${OUTPUT} \
--tgt_angle_1=0.3 \
--tgt_angle_2=-0.26 \
--scale_depth=1 \
--kpt_file_separator=' ' \
--rotate_source
# 왼쪽 오른쪽 0.785가 90도

echo -e '\n rotated_face 만드는중...  \n'

ROTATE=${FACE_FOLDER}/rotating_data
python /home/daehyeon/DepthNets/FaceWarper/warp_dataset.py \
--server_exec /home/daehyeon/DepthNets/FaceWarper/FaceWarperServer/build/FaceWarperServer \
${ROTATE}/ \
--results=${ROTATE}/expected_result/ \
--img_override=${ROTATE}/source/src.png \
--use_dir=affine

# cd ${ROTATE}/expected_result

# ffmpeg -framerate 24 -pattern_type glob -i '*.png' ../../../output.gif

# mv ${ROTATE}/expected_result /home/capstone_ai1/kong/pipeline
