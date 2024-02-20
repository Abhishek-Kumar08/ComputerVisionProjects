import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2

from keras.models import load_model
model = load_model('./Human_Activity_Recognition.h5')

top = tk.Tk()
top.geometry('800x600')
top.title("Human Activity Recognition")
top.configure(background='#CDCDCD')

activity = ['Calling', 'Clapping', 'Cycling', 'Dancing', 'Drinking', 'Eating',
        'Fighting', 'Hugging', 'Laughing', 'Listening to music', 'Running',
        'Sitting', 'Sleeping', 'Texting', 'Using laptop']

def pred_video(file_path):
    cap = cv2.VideoCapture(file_path) 

    if (cap.isOpened()== False): 
        print("Error opening video file") 

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (100, 100) 
    fontScale = 1.5
    color = (255, 150, 50) 
    thickness = 2

    while(cap.isOpened()): 
        
        ret, frame = cap.read() 
        if ret == True: 
            frame = cv2.resize(frame, (1080,720))
            im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            im = cv2.resize(im,(48,48))
            im = im/255
            image = []
            for i in range(21):
                image.append(im)
            image = np.array(image)
            e = model.predict(image)
            e = e[0].argmax()
            e = activity[e]
            frame = cv2.putText(frame, e, org, font, fontScale, color, thickness, cv2.LINE_AA) 
            cv2.imshow('Frame', frame) 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    
        else: 
            break
    
    cap.release() 
    
    cv2.destroyAllWindows() 

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        pred_video(file_path)
    except:
        pass

upload = tk.Button(top, text='Upload a video', command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)

heading = tk.Label(top,text="Human Activity Recognition" ,pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()
