from tkinter import *
from pprint import pprint
import pickle

# Monday to Sunday
schedule_days = 7

# 7AM Start to 9PM
schedule_start_hr = 7.5
schedule_stop_hr = 21
schedule_interval = 1.5
schedule_class_length = int((schedule_stop_hr - schedule_start_hr)/schedule_interval)

# Schedule Array
# CLASS--------
# DAY 
# |
schedule_array = [[" " for _ in range(schedule_class_length)] for _ in range(schedule_days)]

# CREATE GUI
entry_array = [[0 for _ in range(schedule_class_length)] for _ in range(schedule_days)]
pprint(entry_array)

win = Tk()
win.title("Schedule Retriever")

grid_count = 1
for i in range(0,schedule_class_length):
    int_start_hour = int(schedule_start_hr + schedule_interval * i)
    int_start_minute = int((schedule_start_hr + schedule_interval * i - int_start_hour) * 60/10)
    int_stop_hour = int(schedule_start_hr + schedule_interval * (i+1))
    int_stop_minute = int((schedule_start_hr + schedule_interval *(i+1) - int_stop_hour) * 60/10)

    hour = f"{int_start_hour}:{int_start_minute}0 "
    
    if (int_start_hour) < 12: hour += "AM"
    else: hour += "PM"

    hour += f" - {int_stop_hour}:{int_stop_minute}0 "

    if (int_stop_hour) < 12: hour += "AM"
    else: hour += "PM"

    print(hour)
    X = Label(win,text=hour)
    X.grid(row=grid_count, column=0)
    grid_count += 1

# Day Labels
Label(win, text="Monday").grid(row=0, column=1)
Label(win, text="Tuesday").grid(row=0, column=2)
Label(win, text="Wednesday").grid(row=0, column=3)
Label(win, text="Thursday").grid(row=0, column=4)
Label(win, text="Friday").grid(row=0, column=5)
Label(win, text="Saturday").grid(row=0, column=6)
Label(win, text="Sunday").grid(row=0, column=7)



def set_entries():
    row_start = 1
    col_start = 1
    for i in range(schedule_days):
        for j in range(schedule_class_length):
                entry_array[i][j] = Entry(win)
                entry_array[i][j].grid(row=col_start, column=row_start)
                col_start += 1
        row_start += 1
        col_start = 1

def get_entries():
    classes_array = []
    for y in entry_array:
        arr = []
        classes_array.append(arr)
        for x in y:
            arr.append(x.get())

    pprint(classes_array)

    win.destroy()

def load_previous():
    save_file = "saved_schedule.p"
    try:
        test_file = open(save_file, "r")
        entry_array = pickle.load(test_file)

set_entries()

Button(win, text="SET", command=get_entries).grid(row=1, column=8)
Button(win, text="CLEAR", command=set_entries).grid(row=2, column=8)
Button(win, text="LOAD", command=load_previous).grid(row=3, column=8)

win.mainloop()









