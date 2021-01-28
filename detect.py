import cv2 as cv


cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv.INTER_AREA)
    cv.imshow('Input', frame)
    c = cv.waitKey(1)
    if c == 27:
        break
cap.release()
#cv.waitKey()
cv.destroyAllWindows()