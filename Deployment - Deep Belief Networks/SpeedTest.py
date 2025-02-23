import tkinter as tk
import time

root = tk.Tk()
label = tk.Label(root, text="")
label.pack()

def update_time():
    label.config(text=time.strftime("%H:%M:%S.%f"))
    root.after(10, update_time)

update_time()
root.mainloop()