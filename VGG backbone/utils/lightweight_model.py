import tensorflow as tf
from tensorflow.keras import layers, Model
import csv
import numpy as np
from utils.config import cfg

class Mean_Filter(tf.keras.layers.Layer):
    def __init__(self):
        super(Mean_Filter, self).__init__()
    
    def build(self, input_shape):
        filter_shape = (3, 3, input_shape[-1], 1)
        kernel = tf.ones(shape=filter_shape) / 9

        self.mean_filter = lambda x: tf.nn.depthwise_conv2d(
            x, kernel, strides=(1, 1, 1, 1), padding="VALID"
        )
    
    def call(self, x):
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        return self.mean_filter(x)    
    
    
csvfile = open('./utils/vgg_layers.csv', newline='')
vgg_layers = csv.DictReader(csvfile)
new_channels = {3: 3, 64: cfg.new_channel0, 128: cfg.new_channel1, 256: cfg.new_channel2, 512: cfg.new_channel3}  
    
class VggSEncoder(tf.keras.Model):
    def __init__(self, nblocks=3):
        super(VggSEncoder, self).__init__()       
        output_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'][:nblocks+1]
        btnecks = [[] for _ in range(nblocks+1)]

        for idx, vgg_layer in enumerate(vgg_layers):
            name = vgg_layer['name'] 
            typename = vgg_layer['_typename']

            if idx == 0:
                name = 'preprocess' 
                i = 0

            if typename == 'nn.SpatialReflectionPadding':
                btnecks[i].append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')))
            elif typename == 'nn.SpatialConvolution':
                filters = new_channels[int(vgg_layer['filters'])]
                kernel_size = (int(vgg_layer['kernel_size']), int(vgg_layer['kernel_size']))                
                btnecks[i].append(tf.keras.layers.Conv2D(filters, kernel_size, padding='valid', name=name, 
                                                         trainable=False if name == 'preprocess' else True))
            elif typename == 'nn.ReLU':
                btnecks[i].append(tf.keras.layers.ReLU(name=name))                
            elif typename == 'nn.SpatialMaxPooling':
                btnecks[i].append(tf.keras.layers.MaxPooling2D(name=name))
            else:
                raise NotImplementedError(typename)
                
            if name in output_layers:
                i += 1

            if name == output_layers[-1]:
                print("Reached target layer: {}".format(name))
                break        
        
        
        self.btnecks = []
        for i, layers in enumerate(btnecks):
            shape = [3, new_channels[64], new_channels[128], new_channels[256]][i]
            input_tensor = tf.keras.layers.Input(shape=[None, None, shape])
            x = input_tensor  
            mean_input = Mean_Filter()(input_tensor)
            skip = (input_tensor - mean_input) 

            for layer in layers:
                x = layer(x)
            outputs = [x, skip] 
            self.btnecks.append(tf.keras.Model(inputs=input_tensor, outputs=outputs))
        
        initializer = tf.keras.initializers.GlorotUniform()
        self.eigenbases = []
        self.eigenbases.append(tf.Variable(initializer(shape=[64, new_channels[64]], dtype=tf.float32)))
        self.eigenbases.append(tf.Variable(initializer(shape=[128, new_channels[128]], dtype=tf.float32)))
        self.eigenbases.append(tf.Variable(initializer(shape=[256, new_channels[256]], dtype=tf.float32)))
        self.eigenbases.append(tf.Variable(initializer(shape=[512, new_channels[512]], dtype=tf.float32)))

                
    def call(self, block, input_tensor):
        return self.btnecks[block](input_tensor)
    
    def call_branch(self, block, input_tensor):
        x = self.btnecks[block](input_tensor)
        return x[0]
    
       
    
def create_btneck1(bt_layers, n_channels):
    input_layer = layers.Input(shape=(None, None, n_channels[1]))
    skip_layer = layers.Input(shape=(None, None, n_channels[0])) 
    
    x = bt_layers[0](tf.concat([input_layer, skip_layer], -1))
    x = bt_layers[1](x)
    
    return tf.keras.Model(inputs=[input_layer, skip_layer], outputs=x)


def create_btneck3(bt_layers, n_channels):
    input_layer = layers.Input(shape=(None, None, n_channels[1]))
    skip_layer = layers.Input(shape=(None, None, n_channels[0]))

    x = input_layer
    for layer in bt_layers[:-2]:
        x = layer(x) 
    x = x + skip_layer
    x = bt_layers[-2](x)
    x = bt_layers[-1](x)

    return tf.keras.Model(inputs=[input_layer, skip_layer], outputs=x)
    
    
class VggSDecoder(tf.keras.Model):
    def __init__(self):
        super(VggSDecoder, self).__init__()
        self.btnecks = []
        n1 = new_channels[64]
        self.btnecks.append(create_btneck1([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(3, (3,3), padding='valid')], [3, n1]))
        
        n2 = new_channels[128]
        self.btnecks.append(create_btneck3([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n1, (3,3), padding='valid', activation='relu'),
                                            layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n1, (3,3), padding='valid', activation='relu')], [n1, n2]))
        
        n3 = new_channels[256]
        self.btnecks.append(create_btneck3([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n2, (3,3), padding='valid', activation='relu'),
                                            layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n2, (3,3), padding='valid', activation='relu')], [n2, n3]))
        
        n4 = new_channels[512]
        self.btnecks.append(create_btneck3([layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu')], [n3, n4]))

        
    def call(self, block, input_tensor, skip):
        if block != 0:         
            skip = tf.image.resize_with_crop_or_pad(skip, tf.shape(input_tensor)[1] * 2, tf.shape(input_tensor)[2] * 2)
        input_tensor = [input_tensor, skip]
        bt_out = self.btnecks[block](input_tensor)
        return bt_out