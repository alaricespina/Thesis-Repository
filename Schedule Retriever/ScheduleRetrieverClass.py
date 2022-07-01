from tkinter import *
from pprint import pprint
import pickle

class ScheduleRetriever():
    def __init__(self):
        # Monday to Sunday
        self.schedule_days = 7

        # 7AM Start to 9PM
        self.schedule_start_hr = 7.5
        self.schedule_stop_hr = 21
        self.schedule_interval = 1.5
        self.schedule_class_length = int((self.schedule_stop_hr - self.schedule_start_hr)/self.schedule_interval)

        self.save_file = "saved_schedule.p"
        self.createArrays()

    def createArrays(self):
        self.classes_array = []

        # Row - Days
        # Column - Class Hours
        self.entry_array = [[0 for _ in range(self.schedule_class_length)] for _ in range(self.schedule_days)]

    def printArrays(self):
        pprint(self.entry_array)
        pprint(self.classes_array)

    def loadGui(self):
        self.win = Tk()
        self.win.title("Schedule Retriever")
        self.loadScheduleGrid()
        self.loadDayLabels()

    def loadOptions(self):
        Button(self.win, text="CALCULATE", command=self.calculate).grid(row=1, column=8)
        Button(self.win, text="CLEAR", command=self.set_entries).grid(row=2, column=8)
        Button(self.win, text="SAVE", command=self.save_current).grid(row=3, column=8)
        Button(self.win, text="LOAD", command=self.load_previous).grid(row=4, column=8)

    def loadScheduleGrid(self):
        grid_count = 1
        for i in range(0,self.schedule_class_length):
            int_start_hour = int(self.schedule_start_hr + self.schedule_interval * i)
            int_start_minute = int((self.schedule_start_hr + self.schedule_interval * i - int_start_hour) * 60/10)
            int_stop_hour = int(self.schedule_start_hr + self.schedule_interval * (i+1))
            int_stop_minute = int((self.schedule_start_hr + self.schedule_interval *(i+1) - int_stop_hour) * 60/10)

            hour = f"{int_start_hour}:{int_start_minute}0 "
            
            if (int_start_hour) < 12: hour += "AM"
            else: hour += "PM"

            hour += f" - {int_stop_hour}:{int_stop_minute}0 "

            if (int_stop_hour) < 12: hour += "AM"
            else: hour += "PM"

            print(hour)
            X = Label(self.win,text=hour)
            X.grid(row=grid_count, column=0)
            grid_count += 1

    def loadDayLabels(self):
        # Day Labels
        Label(self.win, text="Monday").grid(row=0, column=1)
        Label(self.win, text="Tuesday").grid(row=0, column=2)
        Label(self.win, text="Wednesday").grid(row=0, column=3)
        Label(self.win, text="Thursday").grid(row=0, column=4)
        Label(self.win, text="Friday").grid(row=0, column=5)
        Label(self.win, text="Saturday").grid(row=0, column=6)
        Label(self.win, text="Sunday").grid(row=0, column=7)

    def set_entries(self):
        row_start = 1
        col_start = 1
        for i in range(self.schedule_days):
            for j in range(self.schedule_class_length):
                    self.entry_array[i][j] = Entry(self.win)
                    self.entry_array[i][j].grid(row=col_start, column=row_start)
                    col_start += 1
            row_start += 1
            col_start = 1

    def get_entries(self):
        self.classes_array = []
        for y in self.entry_array:
            arr = []
            self.classes_array.append(arr)
            for x in y:
                arr.append(x.get())

    def load_previous(self): 
        try:
            test_file = open(self.save_file, "rb")
            classes_array = pickle.load(test_file)
            test_file.close()

            row_start = 1
            col_start = 1
            for i in range(self.schedule_days):
                for j in range(self.schedule_class_length):
                        self.entry_array[i][j].grid_forget()
                        self.entry_array[i][j] = Entry(self.win)
                        text=classes_array[i][j]
                        self.entry_array[i][j].delete(0,END)
                        self.entry_array[i][j].insert(0,text)
                        self.entry_array[i][j].grid(row=col_start, column=row_start)
                        col_start += 1
                row_start += 1
                col_start = 1

            print(classes_array)
            print("Loaded")
        except Exception as E:
            test_file = open(self.save_file, "wb")
            test_file.close()
            print(E)

    def save_current(self):
        self.get_entries()
        test_file = open(self.save_file, "wb")
        pickle.dump(self.classes_array, test_file)
        test_file.close()

    def calculate(self):
        self.get_entries()
        
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        results = Toplevel(self.win)

        for index, day in enumerate(self.classes_array):
            res_text = ""
            day_class_count = 0
            max_classes = self.schedule_class_length
            rest_periods = max_classes
            early_class = False

            base_sleep_hour = 6
            early_class_sleep_reduction = 2 

            for classnum, classhour in enumerate(day):
                if classnum == 0 and classhour != "":
                    early_class = True 

                if classhour != "":
                    day_class_count += 1
                    rest_periods -= 1
            
            res_text += f"{days[index]}\n"             
            res_text += f"Resting Periods = {rest_periods}\n"
            res_text += f"Class Periods = {day_class_count}\n"
            res_text += f"Early Class = {'Yes' if early_class else 'No'}\n"
            res_text += f"Base Sleep Hour = {base_sleep_hour}\n"
            res_text += f"Adjusted Sleep Hour = {(base_sleep_hour - early_class_sleep_reduction) if early_class else base_sleep_hour}\n"

            Label(results, text=res_text).pack(side="left")
            results.mainloop()
            
    def run(self):
        self.loadGui()      
        self.set_entries()
        self.loadOptions()
        self.win.mainloop()









