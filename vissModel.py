import VissualizeNN as VisNN
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('saved_model/my_model')
print(model.summary())
sic = []
for layer in model.layers:
    sic.append(layer.input.shape[1])
sic.append(9)

weights = []
for weight in model.weights:
    if "kernel" in weight.name:
        weights.append(weight.numpy())

network = VisNN.DrawNN(sic, weights)
network.draw()