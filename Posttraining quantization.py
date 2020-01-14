import timeit
start1 = timeit.default_timer()
start2 = timeit.default_timer()

# 0. 사용할 패키지 불러오기
import tensorflow as tf
import numpy as np
import pathlib
from numpy import argmax

# 1. 데이터 준비하기
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
#Create the .tflite file

#quantization -미해결
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
mnist_train,_ = mnist.load_data()
images = tf.cast(mnist_train[0], tf.float32) / 255.0
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representative_data_gen():
  for input_value in mnist_ds.take(100):
    yield [input_value]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model_quant_io.tflite"

#3-0 tflite로 convert된 model 사용하기
#load the model into the interpreter
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

#3-1 evaluate the models - quantized && tensorflow lite
prediction_digits = []
for test_image in xhat:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)
Quantized_Accuracy = 0
for i in range(5000):
    if (str(prediction_digits[i]) == str(argmax(y_test[xhat_idx[i]]))):
        Quantized_Accuracy = Quantized_Accuracy+1
print(Quantized_Accuracy/5000)
stop2 = timeit.default_timer()
print(stop2-start2)

#3-1 evaluate the models - nonquantized && tensorflow core
non_quantized_yhat = cloned_model.predict_classes(xhat)
NonQuantized_Accuracy = 0
for i in range(5000):
    if (str(argmax(y_test[xhat_idx[i]])) == str(non_quantized_yhat[i])):
        NonQuantized_Accuracy = NonQuantized_Accuracy+1
print(NonQuantized_Accuracy/5000)

stop1 = timeit.default_timer()
print(stop1 - start1)