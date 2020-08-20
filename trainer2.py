import game

import tensorflow as tf

import itertools
from collections import deque
import copy

import numpy as np # seems unnessecary why not use regular random and skip the import
import random

class TicTacModel: # my class
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(36, activation='relu', input_shape=(27, )),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(9, activation='sigmoid')
        ])
class NeuralTic(): # my class
    def __init__(self, epsilon = 0.7):
        self.cg = game.Game()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(36, activation='relu', input_shape=(27, )),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(9, activation='sigmoid') #softmax alternate
        ])
        self.opt = tf.keras.optimizers.SGD(learning_rate=0.1, name='SGD')

        self.epsilon = epsilon

    def neurMove(self):
        if self.epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < epsilon:
                return random.randrange(9)
        
        q_values = self.model.predict([self.cg.binRead()])
        max_move_index = tf.argmax(q_values, 1)
        return max_move_index
    
    def backProp(self, board, moveInd, target_value):
        output = self.model.predict([self.cg.binRead()])
        target = copy.deepcopy(output)
        target[moveInd] = target_value
        illegal_moves = []
        [illegal_moves.append(x) for x in [0,1,2,3,4,5,6,7,8] if x not in self.cg.possMoves()] # get a list of all illegal moves
        for mi in illegal_move_indexes:
            target[mi] = 0.0
        loss = tf.keras.losses.mean_squared_error(target, output)
        



    def startTrain(numGames = 20000): # my method
        modeler = TicTacModel()
        for gameInd in range(numGames):
            self.cg = game.Game()
            moveTracker = deque()
            while self.cg.winner() is None:
                if self.cg.pturn:
                    indexMove = self.neurMove()
                    moveTracker.appendleft((self.cg.binRead(),indexMove))
                    self.cg.play(indexMove)
                else:
                    # Random availible for now but should maybe switch to q learner later
                    indexMove = random.choice(self.cg.possMoves())
                    self.cg.play(indexMove)
            # reward = 0.0
            win = self.cg.winner()
            if win == 1: #rewards should be tweaked later
                reward = 1.0
            elif win ==  -1:
                reward = 0.0
            elif win == 0:
                reward = 1.0
            next_position, move_index = moveTracker[0]
            self.backProp(next_position, move_index, reward)

def play_training_games(net_context, qplayer, total_games, discount_factor,
                        epsilon, x_strategies, o_strategies): 
                        #x_strategies and o_strategies are how to decide to move typically trained against random opponent but could use Q model
    #converts strartegies to array or something like
    if x_strategies:
        x_strategies_to_use = itertools.cycle(x_strategies)
    if o_strategies:
        o_strategies_to_use = itertools.cycle(o_strategies)

    # repeats 20,000 times
    for game in range(total_games):
        # tracks move history as a deque|not sure why to use deque vs reg tracking|i believe de que is used to keep track of history across methods w/o need for complex returns
        move_history = deque()

        if not x_strategies:
            x = [create_training_player(net_context, move_history, epsilon)]
            x_strategies_to_use = itertools.cycle(x)
        if not o_strategies:
            o = [create_training_player(net_context, move_history, epsilon)]
            o_strategies_to_use = itertools.cycle(o)
        # add on a new strat that is a player ready to be trained and recycle

        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        play_training_game(net_context, move_history, qplayer,
                           x_strategy_to_use, o_strategy_to_use,
                           discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"{game+1}/{total_games} games, using epsilon={epsilon}...")