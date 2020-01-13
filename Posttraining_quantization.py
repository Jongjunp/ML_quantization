import timeit
start = timeit.default_timer()

# 0. 사용할 패키지 불러오기
import tensorflow as tf
import numpy as np
from numpy import argmax

# 1. 실무에 사용할 데이터 준비하기
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 28,28,1).astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(y_test)
xhat_idx = np.random.choice(x_test.shape[0], 5000)
xhat = x_test[xhat_idx]

# 2. 모델 불러오기
model = tf.keras.models.load_model('quantization_practice.h5')
cloned_model = tf.keras.models.clone_model(model)

cloned_model.set_weights(model.get_weights())
# Create the .tflite file

def representative_dataset_gen():
    for i in range(len(xhat)):
        yield xhat[i]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# 3. 모델 사용하기
original_yhat = tflite_model.predict_classes(xhat)
cloned_yhat = cloned_model.predict_classes(xhat)
Original_Accuracy = 0
Cloned_Accuracy = 0
for i in range(5000):
    if (str(argmax(y_test[xhat_idx[i]])) == str(original_yhat[i])):
        Original_Accuracy = Original_Accuracy+1
    if (str(argmax(y_test[xhat_idx[i]])) == str(cloned_yhat[i])):
        Cloned_Accuracy = Cloned_Accuracy+1
print(Original_Accuracy/5000)
print(Cloned_Accuracy/5000)

stop = timeit.default_timer()
print(stop - start)