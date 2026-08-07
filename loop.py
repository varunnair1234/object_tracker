#Loop

import cv2
import numpy as np
from model import AppearanceModel
import torch
from patches import preprocess, generate_candidates

webcam = cv2.VideoCapture(0)        # starts up the webcam of the computer being used

if webcam.isOpened():               # sanity check
    print('yes')

else:
    print('error')
    exit()

for _ in range(10):
    webcam.read()

capture, frame = webcam.read()          # reads first frame of the webcam

if not capture:
    print(f'read failed, capture = {capture}')
    exit()

x, y, w, h = cv2.selectROI('Select Object', frame)          # Returns the top left horizontal and vertical coordinate, along with width and height of rectangle

patch = frame[y:y+h, x:x+w]         # gets the image patch, which is everything within the rectangle

patch_arr = preprocess(patch)

patch_tensor = torch.tensor(patch_arr, dtype = torch.float32)      # we use pytorch so its trained on the GPU and also i've been instructed to use PyTorch

model = AppearanceModel(10)
model.initialize(patch_tensor)

print(model.mean.shape)
print(model.mean.dtype)


#Loop for the webcam, passes through the frames

while True:
    capture, frame = webcam.read()

    if not capture:
        break
    
    #Scoring Section
    candidates = generate_candidates(frame, x, y, w, h)

    scored = []

    for cx, cy, patch in candidates:
        patch = preprocess(patch)
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        score = model.score(patch_tensor)
        scored.append((cx, cy, score, patch_tensor))

    winner = min(scored, key = lambda item : item[2])
    x, y, winner_tensor = winner[0], winner[1], winner[3]

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    model.update(winner_tensor)

    print(len(model.buffer))

    cv2.imshow('Tracker', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


