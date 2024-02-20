import cv2
import numpy as np
from PIL import Image

def get_limits(color):

    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)

    ll = hsvC[0][0][0]-10,100,100
    ul = hsvC[0][0][0]+10,255,255

    ll = np.array(ll, dtype=np.uint8)
    ul = np.array(ul, dtype=np.uint8)

    return ll, ul

color = [0,255,255]

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    hsvImg = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    ll, ul = get_limits(color=color)

    mask = cv2.inRange(hsvImg, ll, ul)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1,y1,x2,y2 = bbox

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

    cv2.imshow('Frame',frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
