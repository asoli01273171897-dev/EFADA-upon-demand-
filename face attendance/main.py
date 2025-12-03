import tkinter as tk
from tkinter import messagebox

import tkinter as tk

def get_button(window, text, color, command, fg='white'):
    button = tk.Button(  
        window,
        text=text,  
        activebackground="black",
        activeforeground="white",
        fg=fg,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )

    return button 

def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text)  # Removed the unnecessary colon
    label.config(font=("sans-serif", 21), justify="left")
    label.grid(row=1, column=0)  # Added grid placement
    return label

def get_entry_text(window):
    inputtxt=tk.text(window,height=2,width=15,font=("Arial",32))
    return inputtxt

def msg_box(title,description):
    messagebox.showinfo(title,description)