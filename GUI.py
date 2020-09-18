import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk , Image
import numpy
from keras.models import load_model
import cv2
from helperfunction import *

Attendencemodel=load_model("AttendenceSystem.h5",custom_objects={'triplet_loss':triplet_loss})

def confidence(passport,selfie,Attendencemodel):
    pimage=cv2.imread(passport)
    simage=cv2.imread(selfie)
    pimg=image_resizing(pimage)
    simg=image_resizing(simage)
    p_encode=img_to_encoding(pimg,Attendencemodel)
    s_encode=img_to_encoding(simg,Attendencemodel)
    dist,conf=computeconfidence(p_encode,s_encode)
    if dist<threshold:
        conf = "Match with confidence: " + str(conf*100)
    else:
        conf = "No match with confidence: "+str(abs(conf*100))
        
    return conf



top = tk.Tk()
top.geometry('800x600')
top.title('Attandence System')
top.resizable(width = True, height = True) 

def open_passport_img():   
    file_path=filedialog.askopenfilename()
    img=face(file_path)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img) 
    panel = Label(top,image= img)
    panel.image = img 
    panel.place(x = 50 , y = 75)
    file_path2=filedialog.askopenfilename()
    img2=face(file_path2)
    img2 = Image.fromarray(img2)
    img2 = ImageTk.PhotoImage(img2)
    panel2 = Label(top,image= img2)
    panel2.image = img2 
    panel2.place(x = 475 , y = 75)
    c = confidence(file_path,file_path2,Attendencemodel)
    text = Label(top,text=c)
    text.place(x = 200 , y = 500 )


    
upload = Button(top,text =  "Upload image",command = open_passport_img,padx=10,pady=5,font=('arial',10,'bold')).place(x = 325, y = 25 , width =175) 

top.mainloop()