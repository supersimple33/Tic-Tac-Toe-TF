# add a header
import copy

class Game:
	def __init__(self, preBoard = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], pTurn = False):
		self.board = copy.deepcopy(preBoard)
		self.pTurn = copy.deepcopy(pTurn)
	
	# argument move is between 0-8. Checks if the move is open/valid.
	#TODO add extremities HERE
	def canPlay(self, move):
		if self.board[move // 3][move % 3] == 0:
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
		ret = ret.replace('-1', 'O')
		ret = ret.replace('1', 'X')
		ret = ret.replace('0', ' ')
		return ret
	
	# backend method to obtain data to feed into a trainer returns board as one array
	def compRead(self):
		return self.board[0] + self.board[1] + self.board[2]
	
	# return code 0 = failed move 1 = move sucess 10-12 = game over and winner announced
	def play(self, move):
		#checks if move valid
		if move < 0 or move > 8:
			# print("Cant move to %d" % move)
			return 0
		elif self.canPlay(move):
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
		res = self.winner()
		if res is not None:
			# print the output of the game
			if res != 0:
				print("Game Over " + str(res) + " Won")
			else:
				print("Tie Game")
			return 11 + res
		return 1
	
	# Checks if there is a winner or tie in the game. returns -1,01
	def winner(self):
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
		for i in range(9):
			if self.canPlay(i):
				return None
		return 0
	
	# Returns an array of all valid moves
	def possMoves(self):
		pMoves = []
		for i in range(9):
			if self.canPlay(i):
				pMoves.append(i)
		return pMoves
	
	def __del__(self):
		del self.board
		del self.pTurn