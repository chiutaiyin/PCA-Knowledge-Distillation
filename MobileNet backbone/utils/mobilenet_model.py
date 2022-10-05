import tensorflow as tf
from tensorflow.keras import layers, Model

class MobilenetEncoder(tf.keras.Model):
    def __init__(self):
        super(MobilenetEncoder, self).__init__()
        mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=[None, None, 3], include_top=False)
        mobilenet.trainable = False
        output_layers = ['conv1_relu', 'conv_dw_2_relu', 'conv_dw_4_relu', 'conv_dw_6_relu']
        btnecks = [[], [], [], []]
        idx = 0
        for layer in mobilenet.layers[1:]:
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
            x = input_tensor
            for layer in layers:
                x = layer(x)
            outputs = [x] 
            self.btnecks.append(tf.keras.Model(inputs=input_tensor, outputs=outputs))
        
    def call(self, block, input_tensor):
        return self.btnecks[block](input_tensor)
    