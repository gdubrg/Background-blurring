import cv2
import time

cap = cv2.VideoCapture(0)


for i in range(0, 30):
    t = time.time()
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    print("FPS: {:.4}".format(1 / (time.time() - t)))

