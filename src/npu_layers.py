import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
from tensorflow import keras
from keras import layers



def npu_dense(npu, layer_name, layer_idx, num_inputs, time_steps, input_size, output_size, w_data, dest_memspace, inputs=None, activation=None, style='normal', last_layer=0):
    SIM_BATCH = 3 # 同时执行的操作数
    BATCH= 6 # 总的batch大小

    # allocate weight matrix
    wdata = np.random.randint(0, 127, size=(output_size, input_size), dtype=np.int8)
    #TODO: use npu.malloc set W


    # Allocate output vectors
    h = 
