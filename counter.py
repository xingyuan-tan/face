import numpy as np
import imutils
import cv2

prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def counter(frame):
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    count = 0

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by requiring a minimum confidence
        if confidence > .1:
            # extract the index of the class label from the detections list
            idx = int(detections[0, 0, i, 1])
            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0))

            count += 1

    return frame, count
