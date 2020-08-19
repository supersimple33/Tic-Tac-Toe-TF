import game
import itertools
from collections import deque

class TicTacModel:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(36, activation='relu', input_shape=(27, )),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(9, activation='sigmoid')
        ])
class NeuralTic(game.Game):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(36, activation='relu', input_shape=(27, )),
            tf.keras.layers.Dense(36, activation='relu'),
            tf.keras.layers.Dense(9, activation='sigmoid')
        ])

def startTrainX(numGames = 20000):
    modeler = TicTacModel()
    for gameInd in range(numGames):
        moveTracker = deque()


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