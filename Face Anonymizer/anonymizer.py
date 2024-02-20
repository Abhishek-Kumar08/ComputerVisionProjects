import cv2
import mediapipe as mp 

mp_face = mp.solutions.face_detection

def process_image(img, fd):
    H,W,_ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = fd.process(img_rgb)

    if out.detections is not None:
        for d in out.detections:
            l_data = d.location_data
            bbox = l_data.relative_bounding_box

            x,y,w,h = bbox.xmin,bbox.ymin,bbox.width, bbox.height

            x = int(x*W)
            w = int(w*W)
            y = int(y*H)
            h = int(h*H)
            try:
                img[y:y+h,x:x+w,:] = cv2.blur(img[y:y+h,x:x+w,:], (30,30))
            except:
                pass

    return img

# img = cv2.imread("Python Programming\Machine Learning\Basics\OpenCV\sample.jpg")

# with mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=0) as fd:
#     img = process_image(img,fd)

# cv2.imshow("img",img)
# cv2.waitKey(0)

# cap = cv2.VideoCapture("Python Programming\Machine Learning\Basics\OpenCV\sample2.mp4")
cap = cv2.VideoCapture(0)
ret = True

while ret:
    ret,frame = cap.read()

    if ret:
        with mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=0) as fd:
            frame = process_image(frame,fd)
        
        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()