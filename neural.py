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

        self.play(tf.argmax(prediction, axis=1).numpy()[0])

    def __init__(self):
        model = tf.keras.models.load_model('saved_model/my_model')
        self.model = model
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # -1, 0, 1 will represent x, blanks, and o
        self.pTurn = False