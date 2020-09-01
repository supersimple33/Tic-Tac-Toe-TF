import game
import copy
# import random
# import sys

# A subclass of game that has an added method for making the best/smartest move

# Simulates algorithms move
def refactoredA(gameObj):
		losses = 0
		for nextMoveA in gameObj.possMoves(): # loop through all the possible moves
			newGameObj = copy.deepcopy(gameObj)
			codeA = newGameObj.play(nextMoveA)
			if codeA == 10:
				continue
			elif codeA != 1:
				continue
			else:
				losses += refactoredB(newGameObj)
		return losses

# Simulates players move
def refactoredB(ghostBoardA):
		losses = 0
		for nextMoveB in ghostBoardA.possMoves():
			ghostBoardB = copy.deepcopy(ghostBoardA)
			codeB = ghostBoardB.play(nextMoveB)
			if codeB == 12: 
				losses += 1
				continue
			elif codeB != 1:
				continue
			else:
				losses += refactoredA(ghostBoardB)
		return losses

class SmartGame(game.Game):
	def smartMove(self):
		moveRepo = {}
		for nextMoveA in self.possMoves():
			losses = 0 # choose the move that has the least number of losses
			ghostBoardA = copy.deepcopy(self)
			codeA = ghostBoardA.play(nextMoveA)
			if codeA == 10: # if a winning move is availible play immediately
				self.play(nextMoveA)
				return
			elif codeA != 1:
				continue
			else:
				losses += refactoredB(ghostBoardA)
			moveRepo[nextMoveA] = losses

		bestMove = min(moveRepo.items(), key=lambda x: x[1])[0]
		self.play(bestMove)

	def __init__(self):
		super().__init__()
	# takes in the game as a argument. Returns how many losses

	def versus(self):
		while self.winner() is None:
			if not self.pTurn:
				move = int(input("Where would you like to go: "))
				self.play(move)
			else:
				self.smartMove()
			print(self.normalDisplay())

if __name__ == '__main__':
	b = SmartGame()
	b.versus()