import inspect
import os

import tensorflow as tf
import numpy as np

checkpoint_dir = 'test1'

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

inputs = tf.keras.Input(shape=(784,), name='digits')
x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # List of metrics to monitor
              metrics=['sparse_categorical_accuracy'])

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

checkpoint_path = os.path.expanduser(current_dir + '/checkpoints/' + checkpoint_dir)

model.load_weights(checkpoint_path)

loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


prediction = model.predict(np.array([test_images[0]]))[0]
print(prediction)
