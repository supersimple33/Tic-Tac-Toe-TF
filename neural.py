import tensorflow as tf
import numpy as np
import game

class NeuralGame(game.Game):
	def neuralMove(self):
		# obtain a prediction for the current board state and play it
		predSet = self.binRead()
		prediction = self.model.predict([predSet])
		# print(prediction.shape)
		# print(prediction)
		# print(tf.argmax(prediction, axis=1).numpy())
		moves = tf.argmax(prediction, axis=1).numpy()
		# print(moves)

		self.play(moves[0])

	# Initialize the model
	def __init__(self):
		super().__init__()
		model = tf.keras.models.load_model('saved_model/my_model')
		self.model = model
		print(self.model.summary())
	
	# Overide method and play against neural network
	def versus(self):
		while self.winner() is None:
			if self.pTurn:
				move = int(input("Where would you like to go: "))
				self.play(move)
			else:
				self.neuralMove()
			print(self.normalDisplay())

if __name__ == '__main__':
	b = NeuralGame()
	b.versus()