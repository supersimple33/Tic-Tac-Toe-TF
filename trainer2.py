import game

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import itertools
from collections import deque
import copy

import numpy as np # seems unnessecary why not use regular random and skip the import
import random

import time

def binConv(compState):
    return [compState[0] == -1, compState[1] == -1, compState[2] == -1, compState[3] == -1, compState[4] == -1, compState[5] == -1, compState[6] == -1, compState[7] == -1, compState[8] == -1, compState[0] == 0, compState[1] == 0, compState[2] == 0, compState[3] == 0, compState[4] == 0, compState[5] == 0, compState[6] == 0, compState[7] == 0, compState[8] == 0, compState[0] == 1, compState[1] == 1, compState[2] == 1, compState[3] == 1, compState[4] == 1, compState[5] == 1, compState[6] == 1, compState[7] == 1, compState[8] == 1]

class NeuralTic(): # my class # is () necessary/what does it do
    def __init__(self, epsilon = 0.75, discountFac = 1.0, lr=0.1):
        self.cg = game.Game()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(36, activation='relu', input_shape=(27, )),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(9, activation='sigmoid') #softmax alternate
        ])

        self.tempoModel = tf.keras.models.clone_model(self.model)

        self.opt = tf.keras.optimizers.SGD(learning_rate=lr, name='SGD')
        self.lossObj = tf.keras.losses.mean_squared_error

        self.avgLoss = tf.keras.metrics.Mean()

        # self.model.build()
        # self.model.compile(optimizer=self.opt, loss=self.lossObj)

        self.epsilon = epsilon
        self.discountFac = discountFac
        # self.train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.lossObj,name='train')

    def neurMove(self):
        if self.epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < self.epsilon:
                return random.randrange(9)
        
        q_values = self.model.predict([self.cg.binRead()])
        max_move_index = tf.argmax(q_values, 1)
        return max_move_index[0].numpy()

    @tf.function(experimental_relax_shapes=True)
    def backProp(self, position, moveInd, target_value):
        feat = tf.convert_to_tensor([binConv(position)])

        illegal_moves = []
        [illegal_moves.append(float(position[x] == 0)) for x in [0,1,2,3,4,5,6,7,8]]
        targetTens = tf.constant(illegal_moves)

        loss = self.backPropHelper(feat, targetTens)
        return loss

    @tf.function(experimental_relax_shapes=True)
    def backPropHelper(self, features, labels):
        # print("Tracing!")
        with tf.GradientTape() as tape:
            # predictions = self.model(features)
            predictions = self.model.predict_step(features)
            loss = self.lossObj(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    # @tf.function(experimental_relax_shapes=True)
    def startTrain(self, numGames = 20000): # my method
        # track how long it takes for each set of epochs and loop through numGames
        curr = time.time()
        for gameInd in range(numGames):
            # initialize a new game for each training step and setup a tracker to track each state and associated move
            self.cg = game.Game()
            moveTracker = deque()
            # keep playing the game unitl somebody wins
            while self.cg.winner() is None:
                if self.cg.pTurn:
                    indexMove = self.neurMove()
                    moveTracker.appendleft((self.cg.compRead(), indexMove))
                else:
                    # should maybe switch to q tabular learner later and see how it work out?
                    # would learning be faster?
                    indexMove = random.choice(self.cg.possMoves())
                ret = self.cg.play(indexMove)
                if ret == 0: # if the choosen move was unplayable exit loop
                    break
            
            self.feedReward(list(moveTracker)) # feed and back prop in the model

            # update the console with stats for the current batch and shorten epsilon
            if (gameInd+1) % (numGames / 20) == 0:
                print(f"{gameInd+1}/{numGames} games, epsilon={round(self.epsilon,2)}...completed in {round(time.time() - curr, 3)}s, L={round(self.avgLoss.result().numpy(), 5)}")
                self.epsilon = max(0, self.epsilon - 0.05) # could also update discount factor
                curr = time.time()
                self.avgLoss.reset_states()

    def feedReward(self, moveTracker):
        # What should be reward for end state
        win = self.cg.winner()
        if win == None:
            reward = 0.0
        elif win == 1:
            reward = 1.0
        elif win ==  -1:
            reward = 0.0
        elif win == 0:
            reward = 1.0
        
        # training
        # create a temporary model to continue to get predictions
        self.tempoModel.set_weights(self.model.get_weights()) 

        # update the model weights for each of the recorded states
        next_position, move_index = moveTracker[0]
        loss = self.backProp(next_position, move_index, reward)
        self.avgLoss.update_state(loss)
        for (position, move_index) in list(moveTracker)[1:] :
            output = self.tempoModel.predict([binConv(next_position)]) # position should hold same formatting as binRead post conversion
            qValue = tf.reduce_max(output) # what was the q value of the last move
            loss = self.backProp(position, move_index, qValue * self.discountFac)
            self.avgLoss.update_state(loss) # track the average loss for stats

            next_position = position

    def record(self):
        print("started saving")
        self.model.save('saved_model/my_model')
        print("finish saved")

trainStation = NeuralTic()
trainStation.startTrain(12000)
trainStation.record()