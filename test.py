
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np


def make_nn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

def dataset():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)



def main():
    model = make_nn()
    with tf.device('/GPU:0'):
        model.fit(dataset(), epochs=1, steps_per_epoch=9)

if __name__ == '__main__':
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    # Check CUDA version
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    print(f"CUDA Version: {cuda_version}")

    # Check cuDNN version
    cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
    print(f"cuDNN Version: {cudnn_version}")
    main()