import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk

import tkinter as tk

def get_button(window, text, bg_color, command, fg='white',):
    button = tk.Button(  
        window,
        text=text,
        bg=bg_color,  
        activebackground="black",
        activeforeground="white",
        fg=fg,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )
    return button
def get_disabled_button(window, text, bg_color, command, fg='white', state=tk.NORMAL):
    button = tk.Button(
        window,
        text=text,
        bg=bg_color,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20),
        state=state  # Corrected the parameter name
    )
    return button

def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text)  # Removed the unnecessary colon
    label.config(font=("sans-serif", 15), justify="left")
    label.grid(row=1, column=0)  # Added grid placement
    return label

def get_entry_text(window):
    inputtxt=tk.Text(window,height=1,width=10,font=("Arial",32))
    return inputtxt

def get_password_entry(window):
    password_entry = tk.Entry(window,show='*', font=("Arial", 32),width=10)
    return password_entry

def msg_box(title,description):
    messagebox.showinfo(title,description)

def get_icon_button(window, image_path, callback, state=tk.DISABLED):
    icon = Image.open(image_path)
    icon_image = ImageTk.PhotoImage(icon)
    button = tk.Button(window, image=icon_image, command=callback, bd=0, relief=tk.FLAT, state=state)
    button.image = icon_image  # Keep a reference to prevent garbage collection
    return button
def get_password_entry2(window):
    password_var = tk.StringVar()
    password_entry = tk.Entry(window, show='*', font=("Arial", 32), width=10, textvariable=password_var)
    return password_entry, password_var

