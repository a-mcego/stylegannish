from __future__ import print_function
from matplotlib import pyplot as plt
from random import choice
import time
import numpy as np
import sys
import os
import cv2
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)

import datasets
import util
import layers

from PIL import Image

if len(sys.argv) != 2:
    print("Give log save folder name as argument.")
    exit(0)

util.set_gpu_settings()

dataset = datasets.Berry("berries128.npy", do_random_flips_rotations=True)
#dataset = datasets.Doom("wholedemo_smaller.npy")
#dataset = datasets.Map("L413.png")
print("Loaded images:",dataset.loaded.shape)

SAVE_TIME = 10 #every SAVE_TIMEth print we also save a checkpoint
SAVE_FOLDER = "saves/"+sys.argv[1]+"/"
PRINTTIME = 30.0 #how often to print status and save test image, in seconds

#-----start hyperparams-----
G_TEST_MOVING_AVERAGE_BETA = 0.999

LATENT_SIZES = [128, 128, 128,  128,  64,   32] #from smallest image to biggest
#LATENT_SIZES = [512, 512, 512, 512, 256, 128,  64,  32,  16] #from the paper, going all the way up to 1024^2

IMAGE_CHANNELS = 3
STYLE_MIX_CHANCE = 0.9

BATCH_SIZE = 16

INTER_LAYER_INTERPOLATION = 'bilinear'
OUTPUT_INTERPOLATION = 'bilinear'
USE_POINTWISE_DENSES = True
#-----end hyperparams-----

#these are calculated using the hyperparams
NUM_LAYERS = len(LATENT_SIZES)
RESIZE_AMOUNTS = [[2**x,2**x] for x in range(NUM_LAYERS-1,-1,-1)]
LAYER_IMAGE_SIZES = [[dataset.get_image_shape()[0]//x[0],dataset.get_image_shape()[1]//x[1]] for x in RESIZE_AMOUNTS]
RESIZE_AMOUNTS[-1] = None

#styleGAN2 style block with weight mod/demod
class StyleBlock(tf.keras.Model):
    def __init__(self,latent_size_prev,latent_size):
        super(StyleBlock, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.std_dense = layers.ScaledDense(latent_size_prev, bias_initializer=tf.ones_initializer())
        self.latent_size = latent_size
        self.conv_outsize = latent_size

        if USE_POINTWISE_DENSES:
            self.D1 = layers.ScaledDense(latent_size*4,activation=self.lrelu)
            self.D2 = layers.ScaledDense(latent_size,activation=self.lrelu)

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

        if USE_POINTWISE_DENSES:
            data = self.D2(self.D1(data))

        return data

#pic out block with weight modulation
class PicOutBlock(tf.keras.Model):
    def __init__(self, image_channels, latent_size):
        super(PicOutBlock, self).__init__()
        self.std_dense = layers.ScaledDense(latent_size, bias_initializer=tf.ones_initializer())
        self.pic_out = layers.ScaledDense(image_channels, use_bias=False)
    
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
            self.upsample = tf.keras.layers.UpSampling2D(interpolation=INTER_LAYER_INTERPOLATION)
            self.styleblock1 = StyleBlock(latent_size_prev,latent_size)
        else:
            self.upsample = None

        self.styleblock2 = StyleBlock(latent_size,latent_size)
        self.pic_out = PicOutBlock(IMAGE_CHANNELS,latent_size)

        if resize is not None:
            self.resizer = tf.keras.layers.UpSampling2D(size=resize, interpolation=OUTPUT_INTERPOLATION)
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
        self.denses = [layers.ScaledDense(latent_size, activation=self.lrelu) for _ in range(n_denses)]
        
    def call(self, data):
        for dense in self.denses:
            data = dense(data)
        return data

class GAN_g(tf.keras.Model):
    def __init__(self):
        super(GAN_g, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.blocks = [
            G_block(LATENT_SIZES[0],LATENT_SIZES[0],LAYER_IMAGE_SIZES[0],False,RESIZE_AMOUNTS[0]), 
            G_block(LATENT_SIZES[0],LATENT_SIZES[1],LAYER_IMAGE_SIZES[1],True,RESIZE_AMOUNTS[1]), 
            G_block(LATENT_SIZES[1],LATENT_SIZES[2],LAYER_IMAGE_SIZES[2],True,RESIZE_AMOUNTS[2]), 
            G_block(LATENT_SIZES[2],LATENT_SIZES[3],LAYER_IMAGE_SIZES[3],True,RESIZE_AMOUNTS[3]),
            G_block(LATENT_SIZES[3],LATENT_SIZES[4],LAYER_IMAGE_SIZES[4],True,RESIZE_AMOUNTS[4]),
            G_block(LATENT_SIZES[4],LATENT_SIZES[5],LAYER_IMAGE_SIZES[5],True,RESIZE_AMOUNTS[5])
        ]
        self.latentmapping = LatentMapping(LATENT_SIZES[0], n_denses=8)

    #def build(self,input_shape):
        #self.dstart = self.add_weight('start_picture',shape=[1,LAYER_IMAGE_SIZES[0][0],LAYER_IMAGE_SIZES[0][1],LATENT_SIZES[0]], initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0), trainable=True)

    def call(self,latent):
        latent = self.latentmapping(latent)

        #dstart = self.dstart
        #newshape = [data.shape[1],dstart.shape[1],dstart.shape[2],dstart.shape[3]]
        #data = tf.broadcast_to(dstart, newshape)

        #data = tf.random.normal(shape=[data.shape[1],LAYER_IMAGE_SIZES[0][0],LAYER_IMAGE_SIZES[0][1],LATENT_SIZES[0]],mean=0.0,stddev=1.0)
        data = tf.zeros(shape=[latent.shape[1],LAYER_IMAGE_SIZES[0][0],LAYER_IMAGE_SIZES[0][1],LATENT_SIZES[0]])

        pics = []
        counter=0
        for b in self.blocks:
            data, pic = b(data, latent[counter])
            pics.append(pic)
            counter += 1

        pics_summed = tf.math.accumulate_n(pics)
        return pics_summed,pics

    #generate *size* vectors of LATENT_SIZES[0]
    #put them on a LATENT_SIZES[0]-dimensional sphere
    def get_random(self,size):
        vec = tf.random.uniform([size,LATENT_SIZES[0]], minval=-1.0, maxval=1.0, dtype=tf.float32)
        vec_len = tf.math.sqrt(tf.reduce_sum(tf.square(vec),axis=-1,keepdims=True))
        vec = tf.multiply(vec,tf.math.reciprocal(vec_len))
        return vec

    def get_random_full(self,size):
        vec = self.get_random(size)
        vec = tf.broadcast_to(vec[tf.newaxis], [NUM_LAYERS,vec.shape[0],vec.shape[1]])
        return vec

class D_block(tf.keras.Model):
    #image_size is size after possible downsampling. make sure it matches!
    def __init__(self, latent_size, downsample, minibatch_stddev=False):
        super(D_block, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)

        self.conv1 = layers.ScaledConv2D(latent_size, activation=self.lrelu,filter_size=3)
        self.conv2 = layers.ScaledConv2D(latent_size, activation=self.lrelu,filter_size=3)

        self.conv_residual = layers.ScaledDense(latent_size, activation=self.lrelu) #equivalent to 1x1 conv
        if downsample:
            self.resizer = tf.keras.layers.AveragePooling2D()
        else:
            self.resizer = None
        self.minibatch_stddev = minibatch_stddev
        self.normalisation_coef = tf.math.rsqrt(2.0)

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

        data = (self.conv2(self.conv1(data))+self.conv_residual(data))*self.normalisation_coef
        if self.resizer is not None:
            data = self.resizer(data)
        return data

class GAN_d(tf.keras.Model):
    def __init__(self):
        super(GAN_d, self).__init__()
        self.lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.2)
        self.cstart = layers.ScaledDense(LATENT_SIZES[-1], activation=self.lrelu) #equivalent to 1x1 conv

        self.blocks = [
            D_block(LATENT_SIZES[-1], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-2], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-3], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-4], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-5], True, minibatch_stddev=True),
            D_block(LATENT_SIZES[-6], False, minibatch_stddev=True)
        ]
        self.cend = layers.ScaledDense(LATENT_SIZES[-6],activation=self.lrelu) #the last layer is basically a dense over the whole thing.
        self.cend2 = layers.ScaledDense(1, use_bias=False)

    def call(self,data):
        data = self.cstart(data)
        for b in self.blocks:
            data = b(data)
        data = tf.reshape(data,[data.shape[0],-1])
        data = self.cend(data)
        data = self.cend2(data)
        return data

generator = GAN_g()
discriminator = GAN_d()
generator_test = GAN_g()
generator_test_initialized = False

optimizer_d = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8, amsgrad=True)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8, amsgrad=True)
optimizer_g_mapping = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.0, beta_2=0.99, epsilon=1e-8, amsgrad=True)

total_updates=0
total_seen=0
frame_n = 0
example_latents = generator.get_random_full(20)

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
        total_updates = util.load_file_to_int(SAVE_FOLDER+"total_updates.txt")
        total_seen = util.load_file_to_int(SAVE_FOLDER+"total_seen.txt")
        frame_n = util.load_file_to_int(SAVE_FOLDER+"frame_n.txt")

        generator(generator.get_random_full(BATCH_SIZE))
        generator_test(generator.get_random_full(BATCH_SIZE)) #have to do this to initialize weights
        generator_test_initialized = True
        example_latents = tf.convert_to_tensor(np.load(SAVE_FOLDER+"example_latents.npy"))

def style_mixing(batch_size, style_mix_chance):
    #output shape: [NUM_LAYERS,BATCH_SIZE,LATENTSIZE]

    rnds = tf.where(tf.random.uniform([batch_size])<style_mix_chance,
                    tf.random.uniform(shape=[batch_size],minval=0,maxval=NUM_LAYERS-1,dtype=tf.int32),
                    tf.constant(NUM_LAYERS,shape=[batch_size],dtype=tf.int32))[tf.newaxis,:]
    rnds = tf.broadcast_to(rnds,shape=[NUM_LAYERS,batch_size])
    ranges = tf.broadcast_to(tf.range(NUM_LAYERS)[:,tf.newaxis],shape=[NUM_LAYERS,batch_size])
    result = tf.cast(rnds<ranges,tf.float32)
    result = tf.stack([1.0-result,result],axis=-2)[:,:,:,tf.newaxis]

    latents = tf.stack([generator.get_random(batch_size),generator.get_random(batch_size)])
    latents = latents[tf.newaxis,:,:,:]
    latents = tf.broadcast_to(latents,[NUM_LAYERS,2,batch_size,latents.shape[-1]])

    latents *= result
    latents = tf.reduce_sum(latents,axis=1)
    return latents

@tf.function
def train():
    data1 = dataset.get_data(BATCH_SIZE)
    rands1 = style_mixing(BATCH_SIZE, STYLE_MIX_CHANCE)
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

    data2 = dataset.get_data(BATCH_SIZE)
    rands2 = style_mixing(BATCH_SIZE, STYLE_MIX_CHANCE)
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

@tf.function
def update_generator_moving_avg(g, g_t):
    zipped = zip(g_t.trainable_variables, g.trainable_variables)
    for z in zipped:
        z[0].assign(z[0]*G_TEST_MOVING_AVERAGE_BETA + z[1]*(1.0-G_TEST_MOVING_AVERAGE_BETA))

while True:
    loss += [n.numpy() for n in train()]

    #do moving average for testing. G_TEST_MOVING_AVERAGE_BETA=0.999 in the paper(s)
    if generator_test_initialized == False:
        #if test generator hasnt been initialized, initialize it now.
        generator_test_initialized = True
        generator_test(generator.get_random_full(BATCH_SIZE))
        zipped = zip(generator_test.trainable_variables, generator.trainable_variables)
        for z in zipped:
            z[0].assign(z[1])
    else:
        #otherwise just update it.
        update_generator_moving_avg(generator, generator_test)

    updates += 1
    seen += BATCH_SIZE

    if time.time()-starttime > PRINTTIME:
        starttime += PRINTTIME
        total_updates += updates
        total_seen += seen

        print(frame_n, total_updates, total_seen, seen/PRINTTIME, loss/seen)
        
        updates = 0
        seen = 0
        loss = np.zeros(shape=(2))

        total, pics = generator_test(example_latents)
        data_gt = util.f32_to_u8(total)
        data_gt = tf.concat(tf.split(data_gt,4,axis=0),axis=-3)
        data_gt = tf.concat(tf.split(data_gt,5,axis=0),axis=-2)
        data_gt = tf.squeeze(data_gt,axis=0)

        data = data_gt.numpy()

        im = Image.fromarray(data)
        im.save(SAVE_FOLDER + "gan/" + str(frame_n) + ".png")

        frame_n += 1

        if frame_n%SAVE_TIME == 0:
            checkpoint = tf.train.Checkpoint(opt_d=optimizer_d, opt_g=optimizer_g, opt_gm=optimizer_g_mapping, gen=generator, disc=discriminator, gen_test=generator_test)
            checkpoint.save(file_prefix=SAVE_FOLDER+'weights/ckpt')
            util.save_int_to_file(SAVE_FOLDER+"total_updates.txt",total_updates)
            util.save_int_to_file(SAVE_FOLDER+"total_seen.txt",total_seen)
            util.save_int_to_file(SAVE_FOLDER+"frame_n.txt", frame_n)
            np.save(SAVE_FOLDER+"example_latents.npy", example_latents.numpy())
            print("checkpoint saved.\r",flush=True,end='')
