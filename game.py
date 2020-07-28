# add a header

class Game:
	def __init__(self):
		self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # -1, 0, 1 will represent x, blanks, and o
		self.pTurn = False
	
	# argument move is between 0-8
	def canPlay(self, move):
		if self.board[move // 3][move % 3] == 0:
			return True
		else:
			return False

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
	
	# returns succ?
	def play(self, move):
		if move < 0 or move > 8:
			print("Cant move to %d" % move)
			return
		if self.canPlay(move):
			self.pTurn = not self.pTurn
			if self.pTurn:
				self.board[move // 3][move % 3] = 1
			else:
				self.board[move // 3][move % 3] = -1
		else:
			print("Cant move to %d" % move)
		res = self.winner()
		if res is not None:
			print("Game Over " + str(res) + " Won")
	
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
		return None
