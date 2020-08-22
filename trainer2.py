import game

import tensorflow as tf

import itertools
from collections import deque
import copy

import numpy as np # seems unnessecary why not use regular random and skip the import
import random

def binConv(compState):
    return [compState[0] == -1, compState[1] == -1, compState[2] == -1, compState[3] == -1, compState[4] == -1, compState[5] == -1, compState[6] == -1, compState[7] == -1, compState[8] == -1, compState[0] == 0, compState[1] == 0, compState[2] == 0, compState[3] == 0, compState[4] == 0, compState[5] == 0, compState[6] == 0, compState[7] == 0, compState[8] == 0, compState[0] == 1, compState[1] == 1, compState[2] == 1, compState[3] == 1, compState[4] == 1, compState[5] == 1, compState[6] == 1, compState[7] == 1, compState[8] == 1]

class NeuralTic(): # my class # is () necessary/what does it do
    def __init__(self, epsilon = 0.7, discountFac = 1.0):
        self.cg = game.Game()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(36, activation='relu', input_shape=(27, )),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(9, activation='sigmoid') #softmax alternate
        ])
        self.opt = tf.keras.optimizers.SGD(learning_rate=0.1, name='SGD')

        self.epsilon = epsilon
        self.discountFac = discountFac

    def neurMove(self):
        if self.epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < epsilon:
                return random.randrange(9)
        
        q_values = self.model.predict([self.cg.binRead()])
        max_move_index = tf.argmax(q_values, 1)
        return max_move_index

    def backProp(self, position, moveInd, target_value, cModel):
        output = self.model.predict([binConv(position)])
        target = copy.deepcopy(output)
        target[moveInd] = target_value
        illegal_moves = []
        [illegal_moves.append(x) for x in [0,1,2,3,4,5,6,7,8] if position[x] != 0]
        for mi in illegal_move_indexes:
            target[mi] = 0.0
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.mean_squared_error(target, output)
            grads = tape.gradient(loss, cModel.trainable_variables) # linking?
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def startTrain(numGames = 20000): # my method
        modeler = TicTacModel()
        wins = 0
        for gameInd in range(numGames):
            self.cg = game.Game()
            moveTracker = deque()
            while self.cg.winner() is None:
                if self.cg.pturn:
                    indexMove = self.neurMove()
                else:
                    # Random availible for now but should maybe switch to q learner later
                    indexMove = random.choice(self.cg.possMoves())
                moveTracker.appendleft((self.cg.compRead(), indexMove))
                self.cg.play(indexMove)
            # reward = 0.0
            win = self.cg.winner()
            if win == 1: #rewards should be tweaked later
                reward = 1.0
                wins += 1
            elif win ==  -1:
                reward = 0.0
            elif win == 0:
                reward = 1.0
            
            # training
            targModel = tf.keras.models.clone_model(self.model) # should be fine to not reset the target model at the end

            next_position, move_index = moveTracker[0]
            self.backProp(next_position, move_index, reward)
            for (position, move_index) in list(moveTracker)[1:] :
                output = targModel.predict([binConv(next_position)]) # position should hold same formatting as binRead post conversion
                qValue = tf.reduce_max(output) # what was the q value of the last move
                self.backProp(position, move_index, qValue * self.discountFac)
                next_position = position
            
            # moving training along
            if (gameInd+1) % (numGames / 10) == 0:
                # self.epsilon = max(0, self.epsilon - 0.1) # could also update discount factor # no epsilon change in training
                print(f"{game+1}/{total_games} games, win percent={(wins * 10) / numGames}, using epsilon={epsilon}...")
                wins = 0

trainStation = NeuralTic()
trainStation.startTrain(1000)