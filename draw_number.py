from board import Board
from math import floor

W_H = 20

def cycle_neighbours(i):
    if i >= 28:
        return 0
    elif i < 0:
        return 28-1
    else:
        return i

def get_neighbours(i, j):
    global board
    neighbours = [[i+1, j], [i-1, j], [i, j+1], [i, j-1], [i+1, j+1], [i-1, j+1], [i+1, j-1], [i-1, j-1]]
    cn = []
    #cycle 
    for coordinates in neighbours:
        coordinates = list(map(cycle_neighbours, coordinates)) # map on list
        cn.append(coordinates)

    return cn 

def draw(event):
    global board
    x = floor(event.x/W_H)
    y = floor(event.y/W_H)
    try:
        board.field[x][y] = 1
        board.draw_coordinate(x, y, "black")
        n = get_neighbours(x, y)
        for i,j in n:
            board.field[x][y] = 1
            board.draw_coordinate(i, j, "black")
        board.platform.update_idletasks()
    except:
        pass

def erase(event=None):
    global board
    board.init_field()
    board.clear()
    board.draw()
    board.platform.update()

def classify(event=None):
    pass
#TODO classifyer

def main():
    global board
    board = Board("NumberClassification", W_H)
    board.platform.bind("<B1-Motion>", draw)
    board.platform.bind("<space>", erase)
    board.platform.bind("<Return>", classify)
    board.platform.focus_set()
    board.start()

main()