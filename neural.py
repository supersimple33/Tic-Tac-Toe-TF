import tensorflow as tf
import numpy as np
import pandas as pd
import game

class NeuralGame(game.Game):
    def neuralMove(self):
        prediction = self.model.predict([self.compRead()])
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