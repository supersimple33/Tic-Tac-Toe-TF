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

gameRepo = pd.DataFrame(columns=['move', "0o", "1o", "2o", "3o", "4o", "5o", "6o", "7o", "8o", "0e", "1e", "2e", "3e", "4e", "5e", "6e", "7e", "8e", "0x", "1x", "2x", "3x", "4x", "5x", "6x", "7x", "8x"])

for i in range(5000):
	g = playGame()
	if g != None:
		for gameState in g.items():
			formedReplay = {"move" : gameState[0], "0o": gameState[1][0] == -1, "1o": gameState[1][1] == -1, "2o": gameState[1][2] == -1, "3o": gameState[1][3] == -1, "4o": gameState[1][4] == -1, "5o": gameState[1][5] == -1, "6o": gameState[1][6] == -1, "7o": gameState[1][7] == -1, "8o": gameState[1][8] == -1, "0e": gameState[1][0] == 0, "1e": gameState[1][1] == 0, "2e": gameState[1][2] == 0, "3e": gameState[1][3] == 0, "4e": gameState[1][4] == 0, "5e": gameState[1][5] == 0, "6e": gameState[1][6] == 0, "7e": gameState[1][7] == 0, "8e": gameState[1][8] == 0, "0x": gameState[1][0] == 1, "1x": gameState[1][1] == 1, "2x": gameState[1][2] == 1, "3x": gameState[1][3] == 1, "4x": gameState[1][4] == 1, "5x": gameState[1][5] == 1, "6x": gameState[1][6] == 1, "7x": gameState[1][7] == 1, "8x": gameState[1][8] == 1}
			# formedReplay = [gameState[0], gameState[1][0], gameState[1][1], gameState[1][2], gameState[1][3], gameState[1][4], gameState[1][5], gameState[1][6], gameState[1][7], gameState[1][8]}
			gameRepo = gameRepo.append(formedReplay, ignore_index=True)
	if i % 1000 == 0:
		print(i)

print("Begin Saving")
# with open("tic.json", 'w') as fp:
# 	json.dump(gameRepo, fp)
# 	print("Saved %d moves to json" % len(gameRepo))

# gameRepo.drop_duplicates(subset = ["0o", "1o", "2o", "3o", "4o", "5o", "6o", "7o", "8o", "0e", "1e", "2e", "3e", "4e", "5e", "6e", "7e", "8e", "0x", "1x", "2x", "3x", "4x", "5x", "6x", "7x", "8x"], keep = 'first', inplace = True)
print(gameRepo)

# pd.sort_values("move", inplace=True)

with open("result.csv", 'w') as fp:
	gameRepo.to_csv(fp, index=False)
	print("Saved %d moves to csv" % (gameRepo.size // 10))

# 13300 at 1000