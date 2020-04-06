from board import Board
from math import floor
from classifier import Classifier

W_H = 20

def cycle_neighbours(i):
    if i >= 28 or i < 0:
        return None
    else:
        return i

def get_neighbours(i, j):
    global board
    neighbours = [(i+1, j), (i-1, j), (i, j+1), (i, j-1), (i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]
    cn = []
    #cycle 
    for coordinates in neighbours:
        x, y = coordinates
        if x >= 28 or x < 0 or y >= 28 or y < 0:
            pass
        else:
            cn.append(coordinates)

    return cn 

def draw(event):
    global board
    x = floor(event.x/W_H)
    y = floor(event.y/W_H)
    try:
        board.field[x][y] = 1
        board.draw_coordinate(x, y, "black")
        #thicker
        n = get_neighbours(x, y)
        for i,j in n:
            board.field[i][j] = 1
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
    global board, classifier
    if classifier == None:
        classifier = Classifier()
        classifier.create()
    classification = classifier.classify(board.field)
    board.draw_output(classification.item())

def main():
    global board, classifier
    board = Board("NumberClassification", W_H)
    board.platform.bind("<B1-Motion>", draw)
    board.platform.bind("<space>", erase)
    board.platform.bind("<Return>", classify)
    board.platform.focus_set()
    classifier = None
    board.start()

main()