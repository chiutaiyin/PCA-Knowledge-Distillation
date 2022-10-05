import tensorflow as tf
from tensorflow.keras import layers, Model
import csv
import numpy as np

csvfile = open('./utils/vgg_layers.csv', newline='')
vgg_layers = csv.DictReader(csvfile)

class VggEncoder(tf.keras.Model):
    def __init__(self, vgg_path, nblocks=3):
        super(VggEncoder, self).__init__()
        output_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'][:nblocks+1]
        btnecks = [[] for _ in range(nblocks+1)]        

        for idx, vgg_layer in enumerate(vgg_layers):
            name = vgg_layer['name'] 
            typename = vgg_layer['_typename']

            if idx == 0:
                name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias
                i = 0

            if typename == 'nn.SpatialReflectionPadding':
                btnecks[i].append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')))
            elif typename == 'nn.SpatialConvolution':
                filters = int(vgg_layer['filters'])
                kernel_size = (int(vgg_layer['kernel_size']), int(vgg_layer['kernel_size']))
                btnecks[i].append(tf.keras.layers.Conv2D(filters, kernel_size, padding='valid', name=name, trainable=False))
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
            shape = [3, 64, 128, 256][i]
            input_tensor = tf.keras.layers.Input(shape=[None, None, shape])
            x = input_tensor  

            for layer in layers:
                x = layer(x)
            outputs = [x]
            self.btnecks.append(tf.keras.Model(inputs=input_tensor, outputs=outputs))
        
        self.load_weights(vgg_path)
        
        
    def call(self, block, input_tensor):
        return self.btnecks[block](input_tensor)
