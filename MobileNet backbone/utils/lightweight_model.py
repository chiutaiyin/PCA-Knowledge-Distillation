import tensorflow as tf
from tensorflow.keras import layers, Model
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

new_layers = {32: cfg.new_channel0, 64: cfg.new_channel1, 128: cfg.new_channel2, 256: cfg.new_channel3}

class MobilenetSEncoder(tf.keras.Model):
    def __init__(self, padding="REFLECT"):
        super(MobilenetSEncoder, self).__init__()
        mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=[224, 224, 3], include_top=False)
        SE_layers = []
        SE_layers.append(tf.keras.layers.InputLayer([None, None, 3]))
        for layer in mobilenet.layers:
            if type(layer) is tf.keras.layers.ZeroPadding2D:
                SE_layers.append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], padding)))
            elif type(layer) is tf.keras.layers.ReLU:
                SE_layers.append(tf.keras.layers.ReLU(name=layer.name))
            elif type(layer) is tf.keras.layers.Conv2D:
                n_filters = new_layers[layer.filters]
                if 'pw' in layer.name:
                    SE_layers.append(tf.keras.layers.Conv2D(n_filters, (1, 1), layer.strides, padding='valid', name=layer.name))
                else:
                    SE_layers.append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], padding)))
                    SE_layers.append(tf.keras.layers.Conv2D(n_filters, (3, 3), layer.strides, padding='valid', name=layer.name))
            elif type(layer) is tf.keras.layers.DepthwiseConv2D:
                if layer.padding == 'same':
                    SE_layers.append(tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)))
                SE_layers.append(tf.keras.layers.DepthwiseConv2D((3,3), layer.strides, padding='valid', name=layer.name))
            if layer.name == 'conv_dw_6_relu':
                break
        self.se_model = tf.keras.Sequential(SE_layers)          
          
        output_layers = ['conv1_relu', 'conv_dw_2_relu', 'conv_dw_4_relu', 'conv_dw_6_relu']
        btnecks = [[], [], [], []]
        idx = 0
        for layer in self.se_model.layers:
            btnecks[idx].append(layer)
            if layer.name == output_layers[idx]:
                idx += 1
            if layer.name == output_layers[-1]:
                break        
        preproc = [tf.keras.layers.Lambda(lambda x: tf.keras.applications.mobilenet.preprocess_input(x * 255.0))]
        btnecks[0] = preproc + btnecks[0]
        
        self.btnecks = []
        for i, layers in enumerate(btnecks):
            shape = layers[1].input_shape[1:] if i == 0 else layers[0].input_shape[1:]
            input_tensor = tf.keras.layers.Input(shape=shape)
            mean_input = Mean_Filter()(input_tensor)
            skip = (input_tensor - mean_input)
            
            x = input_tensor
            for layer in layers:
                x = layer(x)

            outputs = [x, skip] 
            self.btnecks.append(tf.keras.Model(inputs=input_tensor, outputs=outputs))
        
        initializer = tf.keras.initializers.GlorotUniform()
        self.eigenbases = []
        self.eigenbases.append(tf.Variable(initializer(shape=[32, new_layers[32]], dtype=tf.float32)))
        self.eigenbases.append(tf.Variable(initializer(shape=[64, new_layers[64]], dtype=tf.float32)))
        self.eigenbases.append(tf.Variable(initializer(shape=[128, new_layers[128]], dtype=tf.float32)))
        self.eigenbases.append(tf.Variable(initializer(shape=[256, new_layers[256]], dtype=tf.float32)))
                
    def call(self, block, input_tensor):
        return self.btnecks[block](input_tensor)
    
    def call_branch(self, block, input_tensor):
        x = self.btnecks[block](input_tensor)
        return x[0]


def create_btneck1(bt_layers, n_channels):
    input_layer = layers.Input(shape=(None, None, n_channels[1]))
    skip_layer = layers.Input(shape=(None, None, n_channels[0])) 
    
    x = bt_layers[0](input_layer)
    x = bt_layers[1](tf.concat([x, skip_layer], -1))
    x = bt_layers[2](x)
    
    return tf.keras.Model(inputs=[input_layer, skip_layer], outputs=x)


def create_btneck3(bt_layers, n_channels):
    input_layer = layers.Input(shape=(None, None, n_channels[1]))
    skip_layer = layers.Input(shape=(None, None, n_channels[0])) 
    
    x = input_layer
    for layer in bt_layers[:-2]:
        x = layer(x)
    
    x = bt_layers[-2](x + skip_layer)
    x = bt_layers[-1](x)
    
    return tf.keras.Model(inputs=[input_layer, skip_layer], outputs=x)


    
class MobilenetSDecoder(tf.keras.Model):
    def __init__(self, padding="REFLECT"):
        super(MobilenetSDecoder, self).__init__()
        self.btnecks = []
        n1 = new_layers[32]
        self.btnecks.append(create_btneck1([layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(3, (3,3), padding='valid')], [3, n1]))
        n2 = new_layers[64]
        self.btnecks.append(create_btneck3([layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(n1, (3,3), padding='valid', activation='relu'), 
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(n1, (3,3), padding='valid', activation='relu')], [n1, n2]))
        n3 = new_layers[128]
        self.btnecks.append(create_btneck3([layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(n2, (3,3), padding='valid', activation='relu'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(n2, (3,3), padding='valid', activation='relu')], [n2, n3]))
        n4 = new_layers[256]
        self.btnecks.append(create_btneck3([layers.UpSampling2D(interpolation='bilinear'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu'),
                                            layers.Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)),
                                            layers.Conv2D(n3, (3,3), padding='valid', activation='relu')], [n3, n4]))
                        
    def call(self, block, input_tensor, skip):
        skip = tf.image.resize_with_crop_or_pad(skip, tf.shape(input_tensor)[1] * 2, tf.shape(input_tensor)[2] * 2) 
        bt_out = self.btnecks[block]([input_tensor, skip])

        return bt_out