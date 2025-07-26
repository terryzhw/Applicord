import tkinter as tk
from tkinter import ttk
from modules.Data import DataToSheet
from datetime import datetime
from gui.Menu import Menu
from gui.Entry import Entry


class Windows(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Applicord")
        

        # creating frame and assigning container
        container = tk.Frame(self, height=400, width=600)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (Menu, Entry):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Menu)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()






        

    

if __name__ == "__main__":
    Windows().mainloop()