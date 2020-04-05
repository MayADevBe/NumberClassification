import tkinter as tk

class Board:
    '''Creates GUI Board'''

    def __init__(self, title, width):
        self.window = tk.Tk()
        self.window.title(title)
        self.width = width
        self.field = []
        self.init_field()
        self.platform = tk.Canvas(self.window, width = 28*width, height = 28*width)
        self.platform.pack()

    def init_field(self):
        self.field = []
        for i in range(28):
            self.field.append([])
            for j in range(28): 
                self.field[i].append(0)

    def clear(self):
        self.platform.delete("all")

    def draw(self):
        # draw new field + create empty field
        for i in range(28):
            for j in range(28):
                if self.field[i][j] == 1:
                    self.platform.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill="black")
    
    def draw_coordinate(self, i, j, color):
        self.platform.create_rectangle(i*self.width, j*self.width, (i+1)*self.width, (j+1)*self.width, fill=color)

    def draw_output(self, output):
        rndfont = 25
        self.platform.create_text((self.width/2), (self.width/2), text=output, font=('Pursia', rndfont), anchor="center", fill="red", tag="output")

    def start(self):
        self.window.mainloop()

