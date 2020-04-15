import numpy as np
import tensorflow as tf


class Berry:
    def __init__(self, filename, do_random_flips_rotations):
        #load images from numpy package with type uint8 and shape [image,height,width,channel]
        self.loaded = np.load(filename)
        with tf.device('/CPU:0'):
            self.loaded = tf.convert_to_tensor(self.loaded)
        self.do_random_flips_rotations = do_random_flips_rotations
        if self.do_random_flips_rotations:
            with tf.device('/CPU:0'):
                loaded_rot = tf.image.rot90(self.loaded,k=1)
                self.loaded = tf.concat([self.loaded,loaded_rot], axis=0)

    def get_data(self, size):
        data_len = self.loaded.shape[0]
        indices = tf.random.uniform([size],maxval=data_len,dtype=tf.int32)
        data = tf.gather(self.loaded,indices)
        data = tf.cast(data,tf.float32)*(2.0/255.0)-1.0
        if self.do_random_flips_rotations:
            data = tf.image.random_flip_left_right(data)
            data = tf.image.random_flip_up_down(data)
        return data

    def get_image_shape(self):
        return self.loaded.shape[1:]

class Doom:
    def __init__(self,filename):
        #load images from numpy package with type uint8 and shape [image,height,width,channel]
        self.loaded = np.load(filename)
        self.loaded = np.pad(self.loaded, [[0,0],[4,4],[0,0],[0,0]], 'constant', constant_values = 128)
        self.loaded = tf.convert_to_tensor(self.loaded)

    def get_data(self, size):
        data_len = self.loaded.shape[0]
        indices = tf.random.uniform([size],maxval=data_len,dtype=tf.int32)
        data = tf.cast(tf.gather(self.loaded,indices),tf.float32)*(2.0/255.0)-1.0
        return data

    def get_image_shape(self):
        return self.loaded.shape[1:]

class Map:
    def __init__(self, filename):
        self.loaded = tf.io.decode_png(tf.io.read_file(filename))
        self.crop_size = tf.convert_to_tensor([128,128,self.loaded.shape[2]])

    def get_data(self, size):
        imgshape = self.get_image_shape()
        out = []
        for _ in range(size):
            out.append(tf.image.random_crop(self.loaded, self.crop_size))
        out = tf.stack(out,axis=0)
        return tf.cast(out,tf.float32)*(2.0/255.0)-1.0

    def get_image_shape(self):
        return self.crop_size
