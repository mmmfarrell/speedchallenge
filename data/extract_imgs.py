import numpy as np
import cv2
import os
from tqdm import tqdm

# video_file = "./train.mp4"
# dest_dir = "./train/"

video_file = "./test.mp4"
dest_dir = "./test/"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

cap = cv2.VideoCapture(video_file)

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_id in tqdm(range(num_frames)):
    ret, frame = cap.read()

    frame_file = dest_dir + str(frame_id).zfill(5) + ".jpg"
    cv2.imwrite(frame_file, frame)

cap.release()
