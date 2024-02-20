import tkinter as tk
from tkinter import filedialog
from pygame import mixer

import numpy as np
import cv2
import librosa

from keras.models import load_model
model = load_model('./Voice_Gender_Detector.h5')

top = tk.Tk()
top.geometry('800x600')
top.title("Voice Gender Detector")
top.configure(background='#CDCDCD')

file_path = ''

l = tk.Button(top, text='Play', command=lambda: play_audio(file_path), padx=10, pady=5)
label1 = tk.Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))

def play_audio(file_path):
    mixer.init()
    mixer.music.load(file_path)
    mixer.music.play()

def upload_audio():
    try:
        global file_path
        file_path = filedialog.askopenfilename()
        l.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
        label1.configure(text='')
        show_detect_button(file_path)
    except:
        pass

def Detect(file_path):
    global label_packed
    y, sr = librosa.load(file_path, sr=None) 
    d = librosa.stft(y)
    db = librosa.amplitude_to_db(np.abs(d), ref=np.max)
    db = cv2.resize(db,(64,64))
    db = (db+80)/80
    img = []
    for i in range(20):
        img.append(db)
    img = np.array(img)
    e = model.predict(img)
    gen = 'Male' if e[0]<0.5 else 'Female'
    label1.configure(foreground='#011638', text=gen)

def show_detect_button(file_path):
    Detect_b = tk.Button(top, text='Detect Gender', command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

upload = tk.Button(top, text='Upload mp3 file', command=upload_audio, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
l.pack(side='bottom',expand=True)

label1.pack(side="bottom", expand=True)
heading = tk.Label(top,text="Voice Gender Detector" ,pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()
