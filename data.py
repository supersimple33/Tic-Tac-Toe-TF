import game
# import json
import pandas as pd
import random
import copy

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
		# print(move, end = '')
		code = b.play(move)

		if code == 1 and not b.pTurn:
			replay[move] = b.compRead()
		elif code == 10 or code == 11:
			return None
		elif code == 12:
			replay[move] = b.compRead()
			# print(b.normalDisplay())
			return replay
		elif code == 0:
			print("uh oh %d" % move, end = '')

		i += 1
		pmoves.remove(move)
	del b

gameRepo = pd.DataFrame(columns=['move', "0s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s"])

for i in range(10):
	g = playGame()
	if g != None:
		for gameState in g.items():
			formedReplay = {"move" : gameState[0], "0s": gameState[1][0], "1s": gameState[1][1], "2s": gameState[1][2], "3s": gameState[1][3], "4s": gameState[1][4], "5s": gameState[1][5], "6s": gameState[1][6], "7s": gameState[1][7], "8s": gameState[1][8]}
			# formedReplay = [gameState[0], gameState[1][0], gameState[1][1], gameState[1][2], gameState[1][3], gameState[1][4], gameState[1][5], gameState[1][6], gameState[1][7], gameState[1][8]}
			gameRepo = gameRepo.append(formedReplay, ignore_index=True)
	if i % 1000 == 0:
		print(i)

print("Begin Saving")
# with open("tic.json", 'w') as fp:
# 	json.dump(gameRepo, fp)
# 	print("Saved %d moves to json" % len(gameRepo))

gameRepo.drop_duplicates(subset = ["0s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s"], keep = 'first', inplace = True)
print(gameRepo)

# pd.sort_values("move", inplace=True)

with open("result.csv", 'w') as fp:
	gameRepo.to_csv(fp, index=False)
	print("Saved %d moves to csv" % (gameRepo.size // 10))

# 13300 at 1000