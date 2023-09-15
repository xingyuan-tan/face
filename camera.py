import numpy as np
import cv2 as cv
from detector import recognize_faces

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    output = np.array(recognize_faces(frame))

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    # cv_image = cv.cvtColor(np.array(output), cv.COLOR_RGB2RGB)

    # Display the resulting frame
    cv.imshow('frame', output)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

## Test Try #2 hello