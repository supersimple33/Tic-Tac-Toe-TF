import numpy as np
import tensorflow as tf
import pandas as pd

print(tf.test.is_gpu_available())
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with open("result.csv", 'r') as fp:
    df = pd.read_csv(fp)

moves = df.pop("move")
labeledMoves = np.array(moves.values, dtype=np.uint8)
print(labeledMoves.dtype)

dataset = tf.data.Dataset.from_tensor_slices((df.values, labeledMoves))
train_dataset = dataset.shuffle(len(df)).batch(1)

# print(df.head())
# print(df.dtypes)
print(dataset)
# for x,y in dataset:
#     print(x,y)

def get_compiled_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(5, activation='relu'),tf.keras.layers.Dense(3, activation='relu'),tf.keras.layers.Dense(9)])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

model = get_compiled_model()
model.fit(train_dataset, epochs=5)
model.save('saved_model/my_model')
print(model.summary())