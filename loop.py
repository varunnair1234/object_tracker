#Loop

import cv2
import numpy as np
from model import AppearanceModel
import torch

webcam = cv2.VideoCapture(0)        # starts up the webcam of the computer being used

if webcam.isOpened():               # sanity check
    print('yes')

else:
    print('error')
    exit()

for _ in range(10):
    webcam.read()

ret, frame = webcam.read()          # reads first frame of the webcam

if not ret:
    print(f'read failed, ret = {ret}')
    exit()

x, y, w, h = cv2.selectROI('Select Object', frame)          # Returns the top left horizontal and vertical coordinate, along with width and height of rectangle

patch = frame[y:y+h, x:x+w]         # gets the image patch, which is everything within the rectangle

patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)         #   convert to grayscale, simpler model

patch = cv2.resize(patch, (32, 32))             #   resizing to a fixed shape helps PCA because PCA can't work for opposite dimensions
patch_arr = patch.flatten()             #     Flattening the array multilpies the dimensions


print(patch_arr.shape)

patch_tensor = torch.tensor(patch_arr, dtype = torch.float32)      # we use pytorch so its trained on the GPU and also i've been instructed to use PyTorch

model = AppearanceModel(10)
model.initialize(patch_tensor)

print(model.mean.shape)
print(model.mean.dtype)


#Loop for the webcam, passes through the frames

while True:
    ret, frame = webcam.read()

    if not ret:
        break

    cv2.imshow('Tracker', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


