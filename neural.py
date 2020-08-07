import tensorflow as tf
import numpy as np
import pandas as pd
import game

class NeuralGame(game.Game):
    def neuralMove(self):
        state = self.compRead()
        predSet  = [state[0] == -1, state[1] == -1, state[2] == -1, state[3] == -1, state[4] == -1, state[5] == -1, state[6] == -1, state[7] == -1, state[8] == -1, state[0] == 0, state[1] == 0, state[2] == 0, state[3] == 0, state[4] == 0, state[5] == 0, state[6] == 0, state[7] == 0, state[8] == 0, state[0] == 1, state[1] == 1, state[2] == 1, state[3] == 1, state[4] == 1, state[5] == 1, state[6] == 1, state[7] == 1, state[8] == 1]
        prediction = self.model.predict([predSet])
        print(prediction.shape)
        print(prediction)
        print(tf.argmax(prediction, axis=1).numpy())
        moves = tf.argmax(prediction, axis=1).numpy()
        print(moves)

        self.play(moves[0])

    def __init__(self):
        model = tf.keras.models.load_model('saved_model/my_model')
        self.model = model
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # -1, 0, 1 will represent x, blanks, and o
        self.pTurn = False
    
    def versus(self):
        while self.winner() is None:
            if self.pTurn:
                move = int(input("Where would you like to go: "))
                self.play(move)
            else:
                self.neuralMove()
            print(self.normalDisplay())

b = NeuralGame()
b.versus()