# Background blurring for Video call
This is a simple program to blur the background maintaining clean the foreground (the face).
<br>
This is just an exercise useful to review some Computer Vision algorithms.   

## Requirements
* OpenCV 3.3.0
* Numpy 1.13.3

## Usage
Just clone the repo and run the script.
```
python main.py
```
Clearly, a web camera is needed (the program automatically use the port 0 to get images).
Press any key to stop the program and safely exit.

## Implementation
The basic idea is to detect the bounding box of the face, defining a starting 
area for the GrabCut (https://en.wikipedia.org/wiki/GrabCut) algorithm and then blurring the background.

Specifically, main modules are:
* Frame acquisition (*thread*)
* Face Detection
* GrabCut (*thread*)
* Background blurring (*thread*)
* Merge results and show the final frame

### Face Detection
The well-known *Viola&Jones* algorithm is exploited for the Face Detection task. <br> 
It is easily available on OpenCV and it is lightweight. 
* **Input**: acquired frame
* **Output**: *(x, y)* coordinated of the top-right corner of the face bounding box with 
its width and height *(w, h)*
```
faces = face_cascade.detectMultiScale(frame, 1.1, 4)
```

### Face Segmentation
The Grabcut algorithm is used to segment the face, *i.e.* to define the area of the facial pixels.
This algorithm is initialized with a mask containing a portion of the face for sure, *i.e.* the foreground 
pixels and a second area in which pixels belong to the face or to the background.
This first area is obtained making smaller the bounding box of the face.
* **Input**: acquired frame + face bounding box *(x, y, w, h)*
* **Output**: a binary mask with the face segmentation
```
cv2.grabCut(frame, mask, None, self.bgd_model, self.fgd_model, 5, cv2.GC_INIT_WITH_MASK)
```

### Background blurring
The acquired frame is blurred using the blur function of the OpenCV libraries.
A huge filter is here applied in order to have strongly blurred images.
* **Input**: acquired frame
* **Output**: blurred frame
```
frame_blurred = cv2.blur(self.frame, (11, 11))
frame_blurred = cv2.blur(frame_blurred, (21, 21))
```

### Merge results
At this point, the blurred frame and the mask with the face segmentation are ready to be merged.<br>
Facial pixels or blurred pixels are taken relying on the mask information. 
```
frame = np.where((mask[:, :, np.newaxis] == 2) | (mask[:, :, np.newaxis] == 0), frame_blurred, frame_with_face)
```

## Performance
In order to have real-time performance, threads and some tricks are implemented.
* **Threads**: there are 3 different threads in the program. One is for the frame acquisition, one for the
GrabCut algorithm and one for the blurring part. The last two run in parallel, avoiding to sequentially run these two 
independent operations.
```
webcam = Webcam(Queue(1))
webcam.start()

grabcut = GrabCut(frame, x, y, w, h, Queue(1))
grabcut.start()

blur = BlurBkg(Queue(1), frame)
blur.start()
```

* **Queues** are used to transfer data from threads to the main program. The size of all queues is set to 1 
to process always the last acquired frame, skipping the others when needed. 

* **GrabCut** algorithm is computationally heavy. Then, it is run on a resized image 
(a scale factor of 0.3 is tested) and finally the computed mask is upscaled in the original shape.

