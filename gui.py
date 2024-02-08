import tkinter as tk
import librosa.display
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

top = tk.Tk()
top.geometry('800x600')
top.title('Voice Gender Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
model = load_model("results/model.h5")

def extract_feature(audio, sr, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")  
    if chroma or contrast:
        stft = np.abs(librosa.stft(audio))        
    result = np.array([])   
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
        result = np.hstack((result, tonnetz))    
    return result

def detect_gender(file_path):
    global label1, model
    audio, sr = librosa.load(file_path, sr=None)
    features = extract_feature(audio, sr, mel=True).reshape(1, -1)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"  
    print("Detected gender:", gender)  
    label1.configure(foreground="#011638", text='Detected gender: ' + gender)

def show_detect_button(file_path):
    detect_button = Button(top, text="Detect Gender", command=lambda: detect_gender(file_path), padx=10, pady=5)
    detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold')) 
    detect_button.place(relx=0.79, rely=0.46)

def upload_audio():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        label1.configure(text='Uploaded audio file: ' + file_path)
        show_detect_button(file_path)
    except Exception as e:
        print("Error:", e)

upload_button = Button(top, text="Upload Audio", command=upload_audio, padx=10, pady=5)
upload_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload_button.pack(side='bottom', pady=50)

label1.pack(side="bottom", expand=True)
heading = Label(top, text="Voice Gender Detector", pady=20, font=("arial", 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
