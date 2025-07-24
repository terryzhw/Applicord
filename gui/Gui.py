import tkinter as tk
import random

class Windows(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Random Number Generator")
        self.geometry("500x500")
        T = tk.Text(self, height = 5, width = 52)
        l = tk.Label(self, text = "Fact of the Day")
        l.config(font =("Courier", 14))

        Fact = """A man can be arrested in
        Italy for wearing a skirt in public."""

        b1 = tk.Button(self, text = "Next", )
        b2 = tk.Button(self, text = "Exit",
            command = self.destroy) 

        l.pack()
        T.pack()
        b1.pack()
        b2.pack()

        T.insert(tk.END, Fact)

        tk.Button(self, text="Generate", command=self.generate, font=("Arial", 12)).pack(pady=30)

    def generate(self):
        print("Hello")

if __name__ == "__main__":
    Windows().mainloop()
