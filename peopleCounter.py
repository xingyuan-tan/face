# from pyimagesearch.centroidtracker import CentroidTracker
# from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, default=False,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=1,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
# # instantiate our centroid tracker, then initialize a list to store
# # each of our dlib correlation trackers, followed by a dictionary to
# # map each unique object ID to a TrackableObject
# ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# trackers = []
# trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame
    # if we are viewing a video and we did not grab a frame then we have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # if we are supposed to be writing a video to disk, initialize the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    status = "Waiting"
    rects = []
    # check to see if we should run a more computationally expensive object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        # convert the frame to a blob and pass the blob through the network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by requiring a minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the detections list
                idx = int(detections[0, 0, i, 1])
                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                frame = cv2.rectangle(frame,(startX, startY), (endX, endY), (255, 0, 0))

    cv2.imshow('a',frame)
    cv2.waitKey(10)


