import game
import json
import random

#only train when a game results in a win will start with going first
#two diff model one for first one for second maybe?
#SWITCH TO PICKLE after debugging

#returns the replay of good moves NONE if unsuccessful
def playGame():
	i = 0
	replay = {}
	b = game.Game()
	pmoves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	while True: # uh oh the unescable
		# move = random.randint(0, 8)
		move = random.choice(pmoves)
		pmoves.remove(move)
		# print(move, end = '')
		code = b.play(move)

		if code == 1 and not b.pTurn:
			replay[move] = b.compRead()
		elif code == 10 or code == 11:
			return None
		elif code == 12 and b.pTurn:
			replay[move] = b.compRead()
			# print(b.normalDisplay())
			return replay
		elif code == 0:
			print("uh oh %d" % move, end = '')

		i += 1

gameRepo = []

for i in range(100000):
	g = playGame()
	if g != None:
		gameRepo.append(g)

print("Begin Saving")
with open("tic.json", 'w') as fp:
	json.dump(gameRepo, fp)
	print("Saved %d games to json" % len(gameRepo))
	