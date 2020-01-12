import timeit
start = timeit.default_timer()

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(train_images, train_label), (test_images, test_label) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images/255.0, test_images/255.0
train_images = tf.cast(train_images,tf.float32)
test_images = tf.cast(test_images,tf.float32)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_images, train_label, epochs=1)

model.evaluate(test_images,  test_label, verbose=2)

tf.keras.models.save_model(model,'quantization_practice.h5')

model.summary()

stop = timeit.default_timer()
print(stop - start)