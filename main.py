import cv2
import numpy as np
import time
from queue import Queue
from threads import Webcam, BlurBkg, GrabCut


def main():

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    webcam = Webcam(Queue(1))
    webcam.start()

    while True:

        t = time.time()

        frame = webcam.webcam_queue.get()

        # face detection
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)

        # take only one (the first) face
        if False:  # len(faces) > 3
            x, y, w, h = faces[0]

            # grabcut thread
            grabcut = GrabCut(frame, x, y, w, h, Queue(1))
            grabcut.start()

            # blur thread
            blur = BlurBkg(Queue(1), frame)
            blur.start()

            # get thread results
            mask = grabcut.grab_queue.get()
            frame_blurred = blur.blur_queue.get()

            # segment the face only
            frame_with_face = frame * mask[:, :, np.newaxis]

            # merge frame blurred and the face
            frame = np.where((mask[:, :, np.newaxis] == 2) | (mask[:, :, np.newaxis] == 0), frame_blurred, frame_with_face)

            # check face detected
            cv2.circle(frame, (50, 50), 10, (0, 255, 0), -1)

        else:
            # if no face, only blur the frame
            blur = BlurBkg(Queue(1), frame)
            blur.start()
            frame = blur.blur_queue.get()

            # check face detected
            cv2.circle(frame, (50, 50), 10, (0, 0, 255), -1)

        print("FPS: {:.4}".format(1/(time.time() - t)), end="\r")
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k != -1:
            webcam.stp()
            cv2.destroyAllWindows()
            exit()


if __name__ == '__main__':
    main()