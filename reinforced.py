# import tensorflow as tf
# import numpy as np
import pickle
import game
import random
import copy
import sys
import time

class SuperGame:
    def __init__(self, exp_rate=0.3):
        # ML Tweak Vars
        self.exp_rate = exp_rate
        self.lr = 0.15
        self.decay_gamma = 0.7
        # State Saving
        self.states_value1 = {}
        self.states_value2 = {}
        self.states1 = []  # record all positions taken
        self.states2 = []
        self.cg = None
    
    def makeMove(self):
        if random.uniform(0,1) <= self.exp_rate:
            action = random.choice(self.cg.possMoves())
        else:
            value_max = -999 #int.min insteadd?
            for p in self.cg.possMoves(): # loop through every possible move and obtain their q value if there is one
                nextGame = copy.deepcopy(self.cg)
                nextGame.play(p)
                hashBrown = nextGame.getHash()
                if self.cg.pTurn:
                    value = 0 if self.states_value2.get(hashBrown) is None else self.states_value2.get(hashBrown)
                else:
                    value = 0 if self.states_value1.get(hashBrown) is None else self.states_value1.get(hashBrown)
                if value >= value_max: # choose the greates q_value and play it
                    value_max = value
                    action = p
        self.cg.play(action)
        return action

    def trainPlay(self, r=20000):
        # Stats For Tracking Progress
        curr = time.time() # track how long training takes
        ties = 0
        aWins = 0
        bWins = 0

        for i in range(r):
            self.cg = game.Game() # init a new game for each epoch
            esc = False
            if i % 1000 == 0: # update stats every 1000 games
                print("Rounds {} ties percent {}, aw {}, bw {}, durr {}".format(i, ties / 10, aWins / 10, bWins / 10, time.time() - curr))
                ties = 0
                aWins = 0
                bWins = 0
                curr = time.time()
            while not esc: # isEnd?
                # Player 1
                # Make A Move
                p1_action = self.makeMove()

                # Add board state
                hashBrown = self.cg.getHash()
                self.states1.append(hashBrown)

                # check if board status is an end state
                win = self.cg.winner()
                if win is not None:
                    # if ended with p1 either win or draw no other checks needed
                    if win == 0:
                        ties += 1
                    else:
                        aWins += 1

                    # Update Q Tabular Sheet
                    self.feedReward()

                    # exit this epoch and reset
                    esc = True
                    self.states1 = []
                    self.states2 = []
                    break
                else:
                    # Player 2
                    # Make a move
                    p2_action = self.makeMove()

                    # Adds new state
                    hashBrown = self.cg.getHash()
                    self.states2.append(hashBrown)

                    win = self.cg.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw no other checks needed
                        if win == 0:
                            ties += 1
                        else:
                            bWins += 1
                        # Update Q Tabular Sheet
                        self.feedReward()

                        esc = True
                        self.states1 = []
                        self.states2 = []
                        break

    def feedReward(self):
        result = self.cg.winner()
        # determine reward for each based off of result
        if result == 1:
            rewardA = 1.0
            rewardB = -1.0
        elif result == -1:
            rewardA = -1.0
            rewardB = 1.0
        elif result == 0:
            rewardA = 0.0
            rewardB = 0.0
        else:
            sys.exit(1)

        # Add in the reward by updating the q value using a traditional function for updating one
        for st in reversed(self.states1):
            if self.states_value1.get(st) is None:
                self.states_value1[st] = 0
            self.states_value1[st] += self.lr * (self.decay_gamma * rewardA - self.states_value1[st])
            rewardA = self.states_value1[st]
        for st in reversed(self.states2):
            if self.states_value2.get(st) is None:
                self.states_value2[st] = 0
            self.states_value2[st] += self.lr * (self.decay_gamma * rewardB - self.states_value2[st])
            rewardB = self.states_value2[st]

    def save(self):
        print("start saving")
        with open("table1", 'wb') as fp:
            pickle.dump(self.states_value1, fp)
            print("saved 1")
        with open("table2", 'wb') as fp:
            pickle.dump(self.states_value2, fp)
            print("saved 2")

    def load(self):
        print("Start Loading")
        with open("table1", 'rb') as fp:
            self.states_value1 = pickle.load(fp)
            print("Finished loading 1")
        with open("table2", 'rb') as fp:
            self.states_value2 = pickle.load(fp)
            print("Finished loading 2")
    
    def versus(self): # play a HumanReadable game
        self.cg = game.Game()
        while self.cg.winner() is None:
            if not self.cg.pTurn:
                inp = input("Where would you like to go: ")

                move = int(inp)
                self.cg.play(move)
            else:
                self.makeMove()
                # print('a')
            print(self.cg.normalDisplay())
    
    def dataBase(self):
        # print(self.states_value1.keys())
        red = ""
        for keyPair in self.states_value1.items():
            hb = eval(keyPair[0])
            nug = game.Game(preBoard = [[hb[0], hb[1], hb[2]], [hb[3], hb[4], hb[5]], [hb[6], hb[7], hb[8]]])
            # print(nug.normalDisplay())
            # print(round(keyPair[1],11))
            # print()
            red += nug.normalDisplay() + "\n" + str(round(keyPair[1],8)) + "\n" + "\n"
        print("start red save")
        with open("dataBase", 'w') as fp:
            fp.write(red)
            print("finished red save")


# diablo = SuperGame(0.3)
# diablo.load()
# diablo.trainPlay(100000)
# diablo.save()

diablo = SuperGame(0.001)
diablo.load()
diablo.versus()

# diablo = SuperGame(0.001)
# diablo.load()
# diablo.dataBase()

# diablo = SuperGame(0.1)
# diablo.load()
# diablo.trainPlay(5000)
