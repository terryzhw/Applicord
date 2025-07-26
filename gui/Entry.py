import tkinter as tk
from tkinter import ttk
from modules.Data import DataToSheet
from datetime import datetime
from gui.Menu import Menu

class Entry(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.create_entry()
        self.create_button()

    def create_entry(self):
        self.eCompany = tk.Entry(self, width=40)
        self.eCompany.focus_set()
        self.eCompany.pack(pady=10)

        self.ePosition = tk.Entry(self, width=40)
        self.ePosition.focus_set()
        self.ePosition.pack(pady=10)

    def create_button(self):
        ttk.Button(self, text="Add Entry", width=20, command=self.display_text).pack(pady=20)

        back_button = tk.Button(
            self,
            text="Back",
            command=lambda: self.controller.show_frame(Menu),
        )
        back_button.pack(side="bottom", fill=tk.X)

    def display_text(self):
        data = DataToSheet()
        company = self.eCompany.get()
        position = self.ePosition.get()
        date = datetime.today().strftime('%m-%d-%Y')
        status = "Submitted"

        if company == "" or position == "":
            print("incomplete")
        else:
            data.addData(company, position, date, status)
        
        self.eCompany.delete(0, tk.END)
        self.eCompany.insert(0, "")

        self.ePosition.delete(0, tk.END)
        self.ePosition.delete(0, "")