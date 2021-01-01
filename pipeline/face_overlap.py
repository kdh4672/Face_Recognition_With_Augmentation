import cv2
import numpy as np
import argparse

img_list = ['30','37','44','51','65','72','79','85']

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--count', type=int, required=True,
                        help="save img count number")
    args = parser.parse_args()
    return args

args = parse_args()

count = int(args.count)*8
for img_number in img_list:
    background = cv2.imread('./face_alignmented/rotating_data/source/src.png')
    overlay = cv2.imread('./face_alignmented/rotating_data/expected_result/0000{}.png'.format(img_number))
    zeros = np.where(overlay!=0)
    for i in range(len(zeros[0])):
        background[zeros[0][i],zeros[1][i],zeros[2][i]] = 0
    mixed_img = background + overlay
    cv2.imwrite('/database/daehyeon/High_Resolution/augmented/{}.png'.format(count),mixed_img)
    count+=1