#Loop

import cv2
import numpy as np

webcam = cv2.VideoCapture(0)        # starts up the webcam of the computer being used

if webcam.isOpened():               # sanity check
    print('yes')

else:
    print('error')
    exit()

ret, frame = webcam.read()          # reads first frame of the webcam

if not ret:
    print(f'read failed, ret = {ret}')
    exit()

x, y, w, h = cv2.selectROI('Select Object', frame)

patch = frame[y:y+h, x:x+w]

patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

patch = cv2.resize(patch, (32, 32))
patch_arr = patch.flatten()

print(patch_arr.shape)

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


