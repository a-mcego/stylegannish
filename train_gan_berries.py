from __future__ import print_function
import vizdoom as vzd
from matplotlib import pyplot as plt
from random import choice
import time
import numpy as np
import sys
import os
import cv2
import tensorflow as tf
import cv2
#tf.debugging.set_log_device_placement(True)

from PIL import Image

def set_gpu_settings():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

set_gpu_settings()


if len(sys.argv) != 2:
    print("Give log save folder name as argument.")
    exit(0)
    
SAVE_TIME = 1 #every SAVE_TIMEth print we also save a checkpoint
SAVE_FOLDER = "saves/"+sys.argv[1]+"/"
PRINTTIME = 30.0 #how often to print status and save test image, in seconds
G_TEST_MOVING_AVERAGE_BETA = 0.999

LATENT_SIZES = [128, 128, 128,  128,  64,   32] #from smallest image to biggest
IMAGE_CHANNELS = 3
DO_RANDOM_FLIPS = True #flip images randomly up/down and left/right when training

#loaded = np.load("wholedemo_smaller.npy")
loaded = np.load("berries128.npy")
#loaded = tf.convert_to_tensor(loaded[:,:,:,:IMAGE_CHANNELS])
#loaded = np.pad(loaded, [[0,0],[4,4],[0,0],[0,0]], 'constant', constant_values = 128)

print("Loaded images:",loaded.shape)

loaded = tf.convert_to_tensor(loaded)

#batch size
BSIZE = 16

with tf.device('/CPU:0'):
    loaded2 = tf.image.rot90(loaded,k=1)
    loaded = tf.concat([loaded,loaded2], axis=0)

print(loaded.shape)

def to_float(data):
    return tf.cast(data,tf.float32)/(255.0/2.0)-1.0

def to_int(data):
    return tf.cast(tf.clip_by_value((data+1.0)*(255.0/2.0),0.0,255.0),tf.uint8)


class Conv2D3x3(tf.keras.Model):
    def __init__(self, outsize, activation=None):
        super(Conv2D3x3, self).__init__()
        self.conv = tf.keras.layers.Conv2D(outsize, [3,3], strides=[1,1], padding="same", activation=activation)

    def call(self, data):
        return self.conv(data)

class SinCosEmbed(tf.keras.Model):
    def __init__(self):
        super(SinCosEmbed, self).__init__()
        phi = 1.61803398875
        nums = [0,1,2,3,4,5]
        coefs = [4.0*phi**n for n in nums]
        self.coefs = tf.multiply(tf.convert_to_tensor(coefs, dtype=tf.float32),1.57079632679)#constant is pi/2

    def call(self, data):
        out = tf.expand_dims(data,axis=-1)
        out = tf.multiply(out, self.coefs)
        out_sin = tf.math.sin(out)
        out_cos = tf.math.cos(out)
        out = tf.concat([out_sin,out_cos],axis=-1)
        out = tf.reshape(out, tf.concat([out.shape[:-2],[-1]],axis=-1))
        return out

class ScaledDense(tf.keras.Model):
    def __init__(self, outsize, use_bias=True, bias_initializer=tf.zeros_initializer(), activation=None):
        super(ScaledDense, self).__init__()
        self.outsize = outsize
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.insize = input_shape[-1]
        self.dense_weights = self.add_weight('dense_weights',shape=[self.insize,self.outsize], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0), trainable=True)
        self.coef = tf.math.sqrt(2.0 / tf.cast(self.insize,tf.float32))
        if self.use_bias:
            self.bias = self.add_weight('bias',shape=[self.outsize], initializer=self.bias_initializer, trainable=True)
        else:
            self.bias = None
        
    def call(self, data):
        newweights = self.dense_weights * self.coef
        ret = tf.linalg.matmul(data,newweights)
        if self.bias is not None:
            ret = ret + self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        return ret

class ScaledConv2D(tf.keras.Model):
    def __init__(self, outsize, use_bias=True, bias_initializer=tf.zeros_initializer(), activation=None, filter_size=3):
        super(ScaledConv2D, self).__init__()
        self.outsize = outsize
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.filter_size = filter_size

    def build(self, input_shape):
        self.insize = input_shape[-1]
        self.dense_weights = self.add_weight('dense_weights',shape=[self.filter_size, self.filter_size, self.insize, self.outsize], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0), trainable=True)
        self.coef = tf.math.sqrt(2.0 / tf.cast(self.insize * self.filter_size * self.filter_size,tf.float32))
        if self.use_bias:
            self.bias = self.add_weight('bias',shape=[1,1,1,self.outsize], initializer=self.bias_initializer, trainable=True)
        else:
            self.bias = None

    def call(self, data):
        newweights = self.dense_weights * self.coef
        ret = tf.nn.conv2d(data,newweights,strides=[1,1],padding="SAME")
        if self.bias is not None:
            ret = ret + self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        return ret

#a layer that blurs every input feature map separately with a 3x3 filter
class Blur3x3(tf.keras.Model):
    def __init__(self):
        super(Blur3x3, self).__init__()
        self.filter = tf.convert_to_tensor([[1,2,1],[2,4,2],[1,2,1]],dtype=tf.float32)*(1.0/16.0)
        self.filter = self.filter[:,:,tf.newaxis,tf.newaxis]

    def call(self, data):
        ret = tf.transpose(data,[0,3,1,2])
        shape = ret.shape
        ret = tf.reshape(ret, [ret.shape[0]*ret.shape[1],1,ret.shape[2],ret.shape[3]])
        ret = tf.nn.conv2d(ret,self.filter,strides=[1,1],padding="SAME", data_format='NCHW')
        ret = tf.reshape(ret,shape)
        ret = tf.transpose(ret,[0,2,3,1])
        return ret

#DenseLayer = tf.keras.layers.Dense
DenseLayer = ScaledDense

#ConvLayer = Conv2D3x3
ConvLayer = ScaledConv2D

def get_grid(coords, sce):
    return tf.expand_dims(sce(tf.stack(tf.meshgrid(tf.linspace(0.0, 1.0, coords[1]),tf.linspace(0.0, 1.0, coords[0])),axis=-1)),axis=0)

class StyleBlock(tf.keras.Model):
    def __init__(self,latent_size_prev,latent_size):
        super(StyleBlock, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.conv = ConvLayer(latent_size,use_bias=False,filter_size=3)
        self.std_dense = DenseLayer(latent_size_prev, bias_initializer=tf.ones_initializer())
        self.mean_dense = DenseLayer(latent_size_prev)
        self.latent_size = latent_size

    def build(self, input_shape):
        self.bias = self.add_weight('bias_1',shape=[1,1,1,self.latent_size], initializer=tf.zeros_initializer(), trainable=True)
        self.noise_coef = self.add_weight('noise_coef1',shape=[1,1,1,self.latent_size], initializer=tf.zeros_initializer(), trainable=True)

    def adaIN_normalize(self, data):
        std = tf.math.reduce_std(data,axis=[-1,-2],keepdims=True)+1e-5
        mean = tf.math.reduce_mean(data,axis=[-1,-2],keepdims=True)
        data = (data-mean)/std
        return data

    def adaIN_modulate(self, data, latent):
        newstd = self.std_dense(latent)
        newmean = self.mean_dense(latent)

        newshape = [newstd.shape[0],1,1,newstd.shape[1]]
        newstd = tf.reshape(newstd,newshape)
        newmean = tf.reshape(newmean,newshape) #can do this because newstd and newmean have the same shape

        #everything is in the right shape, just put it all together.
        return data*newstd+newmean

    def call(self, data, latent):
        data = self.adaIN_modulate(data, latent)
        data = self.conv(data) 
        data = self.adaIN_normalize(data)
        data += self.bias
        data += tf.multiply(tf.random.normal(data.shape),self.noise_coef)
        data = self.lrelu(data)
        return data

#styleGAN2 style block with weight mod/demod
class StyleBlock2(tf.keras.Model):
    def __init__(self,latent_size_prev,latent_size):
        super(StyleBlock2, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.std_dense = DenseLayer(latent_size_prev, bias_initializer=tf.ones_initializer())
        self.latent_size = latent_size
        self.conv_outsize = latent_size

    def build(self, input_shape):
        self.bias = self.add_weight('styleblock_bias',shape=[1,1,1,self.latent_size], initializer=tf.zeros_initializer(), trainable=True)
        self.noise_coef = self.add_weight('noise_coef',shape=[1,1,1,self.latent_size], initializer=tf.zeros_initializer(), trainable=True)
        self.conv_insize = input_shape[-1]
        self.conv_weights = self.add_weight('conv_weights',shape=[3, 3, self.conv_insize, self.conv_outsize], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0), trainable=True)
        self.conv_coef = tf.math.sqrt(2.0 / tf.cast(self.conv_insize * 3 * 3,tf.float32))

    def call(self, data, latent):
        latent_out = self.std_dense(latent) # [BI] Transform incoming W to style.

        #Modulation:
        data = data*latent_out[:, tf.newaxis, tf.newaxis, :] # [BhwI] Not fused => scale input activations.
        newweights = self.conv_weights * self.conv_coef
        data = tf.nn.conv2d(data, newweights, data_format='NHWC', strides=[1,1,1,1], padding='SAME')
        #Weights are modulated separately
        ww = newweights[tf.newaxis, :, :, :, :]*latent_out[:, tf.newaxis, tf.newaxis, :, tf.newaxis] # [BkkIO] Introduce minibatch dimension. & Scale input feature maps.

        #Demodulation:
        d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.

        data = data*d[:, tf.newaxis, tf.newaxis, :] # [BhwO] Not fused => scale output activations.

        data += self.bias
        data += tf.multiply(tf.random.normal(data.shape),self.noise_coef)
        data = self.lrelu(data)
        return data

class PicOutBlock(tf.keras.Model):
    def __init__(self, image_channels, _):
        super(PicOutBlock, self).__init__()
        self.pic_out = DenseLayer(image_channels, use_bias=False)
    
    def call(self, data, _):
        data = self.pic_out(data)
        return data

#pic out block with weight modulation
class PicOutBlockMod(tf.keras.Model):
    def __init__(self, image_channels, latent_size):
        super(PicOutBlockMod, self).__init__()
        self.std_dense = DenseLayer(latent_size, bias_initializer=tf.ones_initializer())
        self.pic_out = DenseLayer(image_channels, use_bias=False)
    
    def call(self, data, latent):
        latent_out = self.std_dense(latent) # [BI] Transform incoming W to style.
        data = data*latent_out[:, tf.newaxis, tf.newaxis, :]
        data = self.pic_out(data)
        return data


class G_block(tf.keras.Model):
    #image_size is size after possible upsampling. make sure it matches!
    def __init__(self, latent_size_prev, latent_size, image_size, upsample, resize):
        super(G_block, self).__init__()

        self.latent_size = latent_size

        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.image_size = image_size

        if upsample:
            self.upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')
            self.styleblock1 = StyleBlock2(latent_size_prev,latent_size)
        else:
            self.upsample = None

        self.styleblock2 = StyleBlock2(latent_size,latent_size)
        self.pic_out = PicOutBlockMod(IMAGE_CHANNELS,latent_size)

        if resize is not None:
            self.resizer = tf.keras.layers.UpSampling2D(size=resize, interpolation='bilinear')
        else:
            self.resizer = None

    def call(self, data, latent):
        if self.upsample is not None:#hack: upsample being None means we're in the first layer and we shouldn't do styleblock1
            data = self.upsample(data)
            data = self.styleblock1(data,latent)

        data = self.styleblock2(data,latent)

        pic = self.pic_out(data,latent)
        if self.resizer is not None:
            pic = self.resizer(pic)

        return data, pic

class LatentMapping(tf.keras.Model):
    def __init__(self, latent_size, n_denses):
        super(LatentMapping, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.denses = [DenseLayer(latent_size, activation=self.lrelu) for _ in range(n_denses)]
        
    def call(self, data):
        for dense in self.denses:
            data = dense(data)
        return data

class GAN_g(tf.keras.Model):
    def __init__(self):
        super(GAN_g, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.blocks = [
            G_block(LATENT_SIZES[0],LATENT_SIZES[0],[4,4],False,[32,32]), 
            G_block(LATENT_SIZES[0],LATENT_SIZES[1],[8,8],True,[16,16]), 
            G_block(LATENT_SIZES[1],LATENT_SIZES[2],[16,16],True,[8,8]), 
            G_block(LATENT_SIZES[2],LATENT_SIZES[3],[32,32],True,[4,4]),
            G_block(LATENT_SIZES[3],LATENT_SIZES[4],[64,64],True,[2,2]),
            G_block(LATENT_SIZES[4],LATENT_SIZES[5],[128,128],True,None)
        ]
        self.latentmapping = LatentMapping(LATENT_SIZES[0], n_denses=2)

    def build(self,input_shape):
        self.dstart = self.add_weight('start_picture',shape=[1,4,4,LATENT_SIZES[0]], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0), trainable=True)

    def call(self,data):
        latent = self.latentmapping(data)

        newshape = [data.shape[0],4,4,LATENT_SIZES[0]]
        data = tf.broadcast_to(self.dstart, newshape)
        pics = []
        for b in self.blocks:
            data, pic = b(data, latent)
            pics.append(pic)

        pics_summed = tf.math.accumulate_n(pics)
        return pics_summed,pics

    #generate *size* vectors of LATENT_SIZES[0]
    #put them on a LATENT_SIZES[0]-dimensional sphere
    def get_random(self,size):
        vec = tf.random.uniform([size,LATENT_SIZES[0]], minval=-1.0, maxval=1.0, dtype=tf.float32)
        vec_len = tf.math.sqrt(tf.reduce_sum(tf.square(vec),axis=-1,keepdims=True))
        vec = tf.multiply(vec,tf.math.reciprocal(vec_len))
        return vec

class D_block(tf.keras.Model):
    #image_size is size after possible downsampling. make sure it matches!
    def __init__(self, latent_size, downsample, minibatch_stddev=False):
        super(D_block, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)

        self.conv1 = ConvLayer(latent_size, activation=self.lrelu,filter_size=3)
        self.conv2 = ConvLayer(latent_size, activation=self.lrelu,filter_size=3)

        self.conv_residual = DenseLayer(latent_size, activation=self.lrelu) #equivalent to 1x1 conv
        if downsample:
            self.resizer = tf.keras.layers.AveragePooling2D()
        else:
            self.resizer = None
        self.minibatch_stddev = minibatch_stddev

    def do_mb_stddev(self, data):
        y = data-tf.reduce_mean(data, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0, keepdims=True)
        y = tf.sqrt(y + 1e-8) #does this really serve a purpose other than mathematical accuracy?
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)
        y = tf.broadcast_to(y, [data.shape[0], data.shape[1], data.shape[2], 1])
        y = tf.concat([data, y], axis=-1)
        return y

    def call(self, data):
        if self.minibatch_stddev:
            data = self.do_mb_stddev(data)

        data = self.conv2(self.conv1(data))+self.conv_residual(data)
        if self.resizer is not None:
            data = self.resizer(data)
        return data

class GAN_d(tf.keras.Model):
    def __init__(self):
        super(GAN_d, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.cstart = DenseLayer(LATENT_SIZES[-1], activation=self.lrelu) #equivalent to 1x1 conv

        self.blur = Blur3x3()

        self.blocks = [
            D_block(LATENT_SIZES[-1], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-2], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-3], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-4], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-5], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-6], False, minibatch_stddev=True)
        ]
        self.cend = DenseLayer(LATENT_SIZES[-6],activation=self.lrelu) #the last layer is basically a dense over the whole thing.
        self.cend2 = DenseLayer(1, use_bias=False)

    def call(self,data):
        data = self.cstart(data)
        for b in self.blocks:
            #data = self.blur(data)
            data = b(data)
        data = tf.reshape(data,[data.shape[0],-1])
        data = self.cend(data)
        data = self.cend2(data)
        return data

generator = GAN_g()
discriminator = GAN_d()
generator_test = GAN_g()
generator_test_initialized = False

optimizer_d = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8, amsgrad=False)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8, amsgrad=False)
optimizer_g_mapping = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.0, beta_2=0.99, epsilon=1e-8, amsgrad=False)

total_updates=0
total_seen=0
frame_n = 0

def load_file_to_int(filename):
    with open(filename,"r") as file:
        ret = int(file.read())
    return ret

def save_int_to_file(filename,value):
    with open(filename,"w") as file:
        file.write("{0}".format(value))

#make the output directories if they dont exist yet
if not os.path.exists("saves"):
    os.mkdir("saves")
if not os.path.exists(SAVE_FOLDER[:-1]):
    os.mkdir(SAVE_FOLDER[:-1])
    os.mkdir(SAVE_FOLDER+"gan")
else:
    #hack to see if anything was ever saved
    if os.path.isfile(SAVE_FOLDER+"total_updates.txt"):
        checkpoint = tf.train.Checkpoint(opt_d=optimizer_d, opt_g=optimizer_g, opt_gm=optimizer_g_mapping, gen=generator, disc=discriminator, gen_test=generator_test)
        checkpoint.restore(tf.train.latest_checkpoint(SAVE_FOLDER+'weights'))
        total_updates = load_file_to_int(SAVE_FOLDER+"total_updates.txt")
        total_seen = load_file_to_int(SAVE_FOLDER+"total_seen.txt")
        frame_n = load_file_to_int(SAVE_FOLDER+"frame_n.txt")
        generator_test_initialized = True

@tf.function
def train(data1, data2, rands1, rands2):

    data1_fake,_ = generator(rands1)
    with tf.GradientTape() as tape:
        out_real = discriminator(data1)
        out_fake = discriminator(data1_fake)

        #loss A:
        #loss_d = tf.reduce_mean(tf.math.softplus(-out_real))+tf.reduce_mean(tf.math.softplus( out_fake))

        #loss B:
        minus = tf.reduce_mean(out_fake,keepdims=True)-tf.squeeze(out_real,axis=-1)
        loss_d = tf.reduce_mean(tf.math.softplus(minus))

    gradients = tape.gradient(loss_d, discriminator.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, discriminator.trainable_variables))

    with tf.GradientTape() as tape2:
        data2_fake,_ = generator(rands2)
        out_real = discriminator(data2)
        out_fake = discriminator(data2_fake)

        #loss A:
        #loss_g  = tf.reduce_mean(tf.math.softplus( out_real))+tf.reduce_mean(tf.math.softplus(-out_fake))

        #loss B:
        minus = out_fake-tf.squeeze(tf.reduce_mean(out_real,keepdims=True),axis=-1)
        loss_g = tf.reduce_mean(tf.math.softplus(-minus))
    

    gradients = tape2.gradient(loss_g, generator.trainable_variables)
    zipped = zip(gradients, generator.trainable_variables)

    g_mapping_vars = []
    g_other_vars = []

    for v in zipped:
        if v[1].name.startswith('gan_g/latent_mapping'):
            g_mapping_vars.append(v)
        else:
            g_other_vars.append(v)

    optimizer_g.apply_gradients(g_other_vars)
    optimizer_g_mapping.apply_gradients(g_mapping_vars)

    return [tf.reduce_mean(loss_d), tf.reduce_mean(loss_g)]

updates = 0
seen = 0
loss = np.zeros(shape=(2))

starttime = time.time()+5.0

gens = generator.get_random(20)

@tf.function
def get_data():
    data_len = loaded.shape[0]
    indices = tf.random.uniform([BSIZE],maxval=data_len,dtype=tf.int32)
    data = tf.cast(tf.gather(loaded,indices),tf.float32)/(255.0/2.0)-1.0
    if DO_RANDOM_FLIPS:
        data = tf.image.random_flip_left_right(data)
        data = tf.image.random_flip_up_down(data)
    return data

@tf.function
def update_generator_moving_avg(g, g_t):
    zipped = zip(g_t.trainable_variables, g.trainable_variables)
    for z in zipped:
        z[0].assign(z[0]*G_TEST_MOVING_AVERAGE_BETA + z[1]*(1.0-G_TEST_MOVING_AVERAGE_BETA))

while True:
    data = get_data()
    data2 = get_data()

    rands = generator.get_random(BSIZE)
    rands2 = generator.get_random(BSIZE)

    loss += [n.numpy() for n in train(data, data2, rands, rands2)]

    #do moving average for testing. G_TEST_MOVING_AVERAGE_BETA=0.999 in the paper(s)
    if generator_test_initialized == False:
        #if test generator hasnt been initialized, initialize it now.
        generator_test_initialized = True
        generator_test(generator.get_random(BSIZE))
        zipped = zip(generator_test.trainable_variables, generator.trainable_variables)
        for z in zipped:
            z[0].assign(z[1])
    else:
        #otherwise just update it.
        update_generator_moving_avg(generator, generator_test)

    updates += 1
    seen += BSIZE

    if time.time()-starttime > PRINTTIME:
        starttime += PRINTTIME
        total_updates += updates
        total_seen += seen

        print(frame_n, total_updates, total_seen, seen/PRINTTIME, loss/seen)
        
        updates = 0
        seen = 0
        loss = np.zeros(shape=(2))

        if IMAGE_CHANNELS == 3:
            total, pics = generator_test(gens)
            data_gt = to_int(total)
            data_gt = tf.concat(tf.split(data_gt,4,axis=0),axis=-3)
            data_gt = tf.concat(tf.split(data_gt,5,axis=0),axis=-2)
            data_gt = tf.squeeze(data_gt,axis=0)

            data = data_gt.numpy()

        elif IMAGE_CHANNELS == 5:
            rands = generator.get_random(4)

            data = generator(rands)
            data = to_int(data)
            data = tf.concat([data[0],data[1],data[2],data[3]],axis=-2)

            color = data[:,:,0:3]
            depth = data[:,:,3:4]
            rects = data[:,:,4:5]

            depth = np.broadcast_to(depth, color.shape)
            rects = np.broadcast_to(rects, color.shape)

            data = np.concatenate([color,depth,rects],axis=0)

        im = Image.fromarray(data)
        im.save(SAVE_FOLDER + "gan/" + str(frame_n) + ".png")

        frame_n += 1

        if frame_n%SAVE_TIME == 0:
            checkpoint = tf.train.Checkpoint(opt_d=optimizer_d, opt_g=optimizer_g, opt_gm=optimizer_g_mapping, gen=generator, disc=discriminator, gen_test=generator_test)
            checkpoint.save(file_prefix=SAVE_FOLDER+'weights/ckpt')
            save_int_to_file(SAVE_FOLDER+"total_updates.txt",total_updates)
            save_int_to_file(SAVE_FOLDER+"total_seen.txt",total_seen)
            save_int_to_file(SAVE_FOLDER+"frame_n.txt", frame_n)
            print("checkpoint saved.\r",flush=True,end='')
