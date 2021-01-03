for var in {1..11}
do
    echo $var 번째 사진 처리중..
    bash face_augmentation_copy.sh $var
    python face_overlap.py \
    --count=$var
    # 이제 여기다가 각도별 겹쳐서 저장하는 파이썬 파일써야함...
done