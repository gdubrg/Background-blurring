import cv2
import numpy as np
import threading


class Webcam(threading.Thread):

    def __init__(self, webcam_queue):

        threading.Thread.__init__(self)
        self.webcam_queue = webcam_queue
        self.cap = cv2.VideoCapture(0)
        self.stop = False

    def run(self):

        while True and not self.stop:
            ret, frame = self.cap.read()

            if ret:
                img = frame
            else:
                img = np.zeros((480, 640, 3), dtype=np.uint8)

            self.webcam_queue.queue.clear()
            self.webcam_queue.put(img)

    def stp(self):
        self.stop = True


class BlurBkg(threading.Thread):

    def __init__(self, blur_queue, frame):

        threading.Thread.__init__(self)
        self.blur_queue = blur_queue
        self.frame = frame

    def run(self):

        frame_blurred = cv2.blur(self.frame, (11, 11))
        frame_blurred = cv2.blur(frame_blurred, (21, 21))
        self.blur_queue.queue.clear()
        self.blur_queue.put(frame_blurred)


class GrabCut(threading.Thread):
    def __init__(self, frame, x, y, w, h, scale, grab_queue):
        threading.Thread.__init__(self)

        self.frame = frame
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.scale = scale
        self.grab_queue = grab_queue

        self.bgd_model = np.zeros((1, 65), np.float64)
        self.fgd_model = np.zeros((1, 65), np.float64)

    def run(self):
        original_shape = self.frame.shape
        frame = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale)
        frame = cv2.blur(frame, (5, 5))
        mask = np.zeros(frame.shape[:2], np.uint8)

        x = int(self.x * self.scale)
        y = int(self.y * self.scale)
        w = int(self.w * self.scale)
        h = int(self.h * self.scale)

        offset_x = int((w * 0.7) * self.scale)
        offset_y = int((h * 0.7) * self.scale)

        x = x + int(w / 2)
        y = y + int(h / 2)

        mask[y - offset_y:y + offset_y, x - offset_x:x + offset_x] = cv2.GC_FGD
        mask[mask != cv2.GC_FGD] = cv2.GC_PR_BGD

        cv2.grabCut(frame, mask, None, self.bgd_model, self.fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        mask_final = cv2.resize(mask_final, dsize=(original_shape[1], original_shape[0]))

        self.grab_queue.queue.clear()
        self.grab_queue.put(mask_final)

