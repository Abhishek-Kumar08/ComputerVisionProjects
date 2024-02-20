import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

from keras.models import load_model
model = load_model('Python Programming\Machine Learning\Models\Human Activity Recognition\Human_Activity_Recognition.h5')

top = tk.Tk()
top.geometry('800x600')
top.title("Human Activity Recognition")
top.configure(background='#CDCDCD')

label = tk.Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = tk.Label(top)

activity = ['Calling', 'Clapping', 'Cycling', 'Dancing', 'Drinking', 'Eating',
        'Fighting', 'Hugging', 'Laughing', 'Listening to music', 'Running',
        'Sitting', 'Sleeping', 'Texting', 'Using laptop']

def Detect(file_path):
    global label_packed
    im = cv2.imread(file_path,0)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(48,48))
    im = im/255
    image = []
    for i in range(21):
        image.append(im)
    image = np.array(image)
    e = model.predict(image)
    e = e[0].argmax()
    e = activity[e]
    label.configure(foreground='#011638', text=e)

def show_detect_button(file_path):
    Detect_b = tk.Button(top, text='Detect Image', command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_detect_button(file_path)
    
    except:
        pass

upload = tk.Button(top, text='Upload an image', command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand=True)

label.pack(side="bottom", expand=True)
heading = tk.Label(top,text="Human Activity Recognition" ,pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
