#Loop

import cv2

webcam = cv2.VideoCapture(0)

if webcam.isOpened():
    print('yes')

else:
    print(f'error')
    exit()

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


