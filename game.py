# add a header
import copy

class Game:
	def __init__(self, preBoard = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], pTurn = False):
		self.board = copy.deepcopy(preBoard)
		self.pTurn = copy.deepcopy(pTurn)
		self.recentWinner = None
		self.recentPossMoves = [0,1,2,3,4,5,6,7,8]

	# argument move is between 0-8. Checks if the move is open/valid.
	#TODO add extremities HERE
	def __canPlay(self, move):
		if move > 8 or move < 0:
			return False
		elif self.board[move // 3][move % 3] == 0:
			return True
		else:
			return False

	#Displays the board with all the current moves represented by numbers
	def display(self):
		i = 0
		ret = ""
		for row in self.board:
			line = '|'.join(str(s).center(4) for s in row)
			ret += line + "\n"
			i += 1
			if i != 3:
				ret += '-' * len(line) + "\n"
		return ret

	# prints a regular tic tac toe board with xs os and blanks
	def normalDisplay(self):
		ret = self.display()
		ret = ret.replace('-1', 'O ')
		ret = ret.replace('1', 'X')
		ret = ret.replace('0', ' ')
		return ret

	# backend method to obtain data to feed into a trainer returns board as one array
	def compRead(self):
		return self.board[0] + self.board[1] + self.board[2]

	def binRead(self):
		state = self.compRead()
		return [state[0] == -1, state[1] == -1, state[2] == -1, state[3] == -1, state[4] == -1, state[5] == -1, state[6] == -1, state[7] == -1, state[8] == -1, state[0] == 0, state[1] == 0, state[2] == 0, state[3] == 0, state[4] == 0, state[5] == 0, state[6] == 0, state[7] == 0, state[8] == 0, state[0] == 1, state[1] == 1, state[2] == 1, state[3] == 1, state[4] == 1, state[5] == 1, state[6] == 1, state[7] == 1, state[8] == 1]

	# return code 0 = failed move 1 = move sucess 10-12 = game over and winner announced
	def play(self, move):
		#checks if move valid
		if self.__canPlay(move):
			#if move is valid flip whos turn it is and add the move to the board
			self.pTurn = not self.pTurn
			if self.pTurn:
				self.board[move // 3][move % 3] = 1
			else:
				self.board[move // 3][move % 3] = -1
		else:
			# print("Cant move to %d" % move)
			return 0
		# check if the move resulted in a winner
		poss = self.__updatePossMoves()
		self.recentPossMoves = poss
		res = self.__updateWinner()
		self.recentWinner = res
		if res is not None:
			# print the output of the game
			# if res != 0:
			# 	# print("Game Over " + str(res) + " Won")
			# 	pass
			# else:
			# 	# print("Tie Game")
			# 	pass
			return 11 + res
		return 1

	def __updateWinner(self):
		for i in range(3):
			if self.board[i][0] == 0:
				continue
			elif self.board[i][0] == self.board[i][1] and self.board[i][0] == self.board[i][2]:
				return self.board[i][0]
		for i in range(3):
			if self.board[0][i] == 0:
				continue
			elif self.board[0][i] == self.board[1][i] and self.board[0][i] == self.board[2][i]:
				return self.board[0][i]
		if self.board[0][0] == self.board[1][1] and self.board[0][0] == self.board[2][2]:
			if not self.board[0][0] == 0:
				return self.board[0][0]
		if self.board[2][0] == self.board[1][1] and self.board[0][2] == self.board[1][1]:
			if not self.board[1][1] == 0:
				return self.board[1][1]
		if len(self.recentPossMoves) > 0:
			return None
		return 0

	# Checks if there is a winner or tie in the game. returns -1,01
	def winner(self): # method should be removed
		return self.recentWinner

	def __updatePossMoves(self):
		pMoves = []
		for i in range(9):
			if self.__canPlay(i):
				pMoves.append(i)
		return pMoves

	# Returns an array of all valid moves
	def possMoves(self): # method should be removed in favor of get vars
		return self.recentPossMoves

	def __del__(self):
		del self.board
		del self.pTurn

	# returns a unique string linked to the current board state
	def getHash(self):
		return str(self.compRead())
	
	# Play a Player vs Player game
	def versus(self):
		while self.winner() is None:
				if self.pTurn:
					move = int(input("Where would you like to go O: "))
					self.play(move)
				else:
					move = int(input("Where would you like to go X: "))
					self.play(move)
				print(self.normalDisplay())

if __name__ == '__main__':
	g = Game()
	g.versus()