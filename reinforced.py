import tensorflow as tf
import numpy as np
import pickle
import game
import random
import copy
import sys

class SuperGame():
    def __init__(self, exp_rate=0.3):
        # ML Tweak Vars
        self.exp_rate = exp_rate
        self.lr = 0.4
        self.decay_gamma = 0.4
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
            for p in self.cg.possMoves():
                nextGame = copy.deepcopy(self.cg)
                nextGame.play(p)
                hashBrown = nextGame.getHash()
                if self.cg.pTurn:
                    value = 0 if self.states_value2.get(hashBrown) is None else self.states_value2.get(hashBrown)
                else:
                    value = 0 if self.states_value1.get(hashBrown) is None else self.states_value1.get(hashBrown)
                if value >= value_max:
                    value_max = value
                    action = p
            # print(value_max)
        # print("I think " + str(action))
        self.cg.play(action)
        return action

    def trainPlay(self, r=20000):
            for i in range(r):
                self.cg = game.Game()
                esc = False
                if i % 1000 == 0:
                    print("Rounds {}".format(i))
                while not esc: # isEnd?
                    # Player 1
                    # positions = g.possMoves()
                    # Make A Move
                    p1_action = self.makeMove()

                    # Add board state
                    hashBrown = self.cg.getHash()
                    self.states1.append(hashBrown)
                    # check board status if it is end

                    win = self.cg.winner()
                    if win is not None:
                        # self.showBoard()
                        # ended with p1 either win or draw
                        self.feedReward()
                        # self.p1.reset()
                        # self.p2.reset()
                        esc = True
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
                            # ended with p2 either win or draw
                            self.feedReward()
                            esc = True
                            break

    def feedReward(self):
        result = self.cg.winner()
        # backpropagate reward
        if result == 1:
            rewardA = 1
            rewardB = 0
        elif result == -1:
            rewardA = 0
            rewardB = 1
        elif result == 0:
            rewardA = 0.1
            rewardB = 0.5
        else:
            sys.exit(1)

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
        with open("table2", 'rb') as fp:
            self.states_value2 = pickle.load(fp)
            print("Finished loading")
    
    def versus(self):
        self.cg = game.Game()
        while self.cg.winner() is None:
            if self.cg.pTurn:
                move = int(input("Where would you like to go: "))
                self.cg.play(move)
            else:
                self.makeMove()
            print(self.cg.normalDisplay())

diablo = SuperGame(0.01)
diablo.load()
# diablo.trainPlay()
# diablo.save()
diablo.versus()

