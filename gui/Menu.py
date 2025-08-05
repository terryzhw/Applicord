import tkinter as tk
from tkinter import ttk

class Menu(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Menu")
        label.pack(padx=10, pady=10)
        self.controller = controller

        from gui.entry import Entry

        switch_entry_button = tk.Button(
            self,
            text="To Entry",
            command=lambda: self.controller.show_frame(Entry),
        )
        switch_entry_button.pack(side="bottom", fill=tk.X)
    
