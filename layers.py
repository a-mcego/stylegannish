import tensorflow as tf

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