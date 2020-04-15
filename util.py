from __future__ import print_function
from matplotlib import pyplot as plt
from random import choice
import numpy as np
import tensorflow as tf

#cast from uint8[0,255] to float32[-1,1]
def u8_to_f32(data):
    return tf.cast(data,tf.float32)*(2.0/255.0)-1.0

#cast from float32[-1,1] to uint8[0,255]
def f32_to_u8(data):
    return tf.cast(tf.clip_by_value((data+1.0)*(255.0/2.0),0.0,255.0),tf.uint8)

def set_gpu_settings():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_file_to_int(filename):
    with open(filename,"r") as file:
        ret = int(file.read())
    return ret

def save_int_to_file(filename,value):
    with open(filename,"w") as file:
        file.write("{0}".format(value))
