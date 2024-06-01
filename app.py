import tkinter as tk
from tkinter import filedialog, Text
import speech_recognition as sr
from gtts import gTTS
import pytesseract
from PIL import Image
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to handle speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            display_text.insert(tk.END, "Listening...\n")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            display_text.insert(tk.END, f"Recognized Speech: {text}\n")
        except sr.UnknownValueError:
            display_text.insert(tk.END, "Speech recognition could not understand audio\n")
        except sr.RequestError:
            display_text.insert(tk.END, "Could not request results from Google Speech Recognition service\n")
    except Exception as e:
        display_text.insert(tk.END, f"Microphone error: {str(e)}\n")

# Function to handle text-to-speech
def text_to_speech():
    text = input_text.get("1.0", tk.END).strip()
    if text:
        tts = gTTS(text)
        tts.save("output.mp3")
        os.system("start output.mp3")

# Function to handle image recognition (OCR)
def recognize_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        text = pytesseract.image_to_string(Image.open(file_path))
        display_text.insert(tk.END, f"Recognized Text from Image: {text}\n")

# Function to classify image using a pre-trained model
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        display_text.insert(tk.END, "Image Classification:\n")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            display_text.insert(tk.END, f"{i+1}: {label} ({score:.2f})\n")

# GUI Setup
root = tk.Tk()
root.title("Accessibility Tools")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

# Buttons
speech_button = tk.Button(frame, text="Speech Recognition", command=recognize_speech)
speech_button.grid(row=0, column=0, padx=10, pady=10)

text_button = tk.Button(frame, text="Text-to-Speech", command=text_to_speech)
text_button.grid(row=0, column=1, padx=10, pady=10)

ocr_button = tk.Button(frame, text="Recognize Image Text", command=recognize_image)
ocr_button.grid(row=0, column=2, padx=10, pady=10)

classify_button = tk.Button(frame, text="Classify Image", command=classify_image)
classify_button.grid(row=0, column=3, padx=10, pady=10)

# Text areas
input_text = Text(frame, height=5, width=70, wrap='word')
input_text.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

display_text = Text(frame, height=10, width=70, wrap='word')
display_text.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

root.mainloop()
