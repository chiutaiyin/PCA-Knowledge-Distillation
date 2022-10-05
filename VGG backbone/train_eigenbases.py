import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from utils.vgg_model import VggEncoder
from utils.lightweight_model import VggSEncoder
from utils.DatasetAPI import mscoco_dataset
from utils.config import cfg
import argparse
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

mse = tf.keras.losses.MeanSquaredError()
def train(vgg_encoder, vgg_sencoder, input_img):
    losses = []        

    x = input_img
    for l in range(4):
        x = vgg_encoder(l, x)

        m1 = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        feat_c = x-m1
        down = tf.linalg.einsum('bijc,cd->bijd', feat_c, vgg_sencoder.eigenbases[l])
        up = tf.linalg.einsum('bijd,cd->bijc', down, vgg_sencoder.eigenbases[l])
        losses.append(mse(up, feat_c))

        vt_v = tf.matmul(vgg_sencoder.eigenbases[l], vgg_sencoder.eigenbases[l],  transpose_a=True)
        losses.append(mse(vt_v, tf.eye(tf.shape(vt_v)[0])))

    return losses
            

    
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', '-i', type=int, default=10, 
                    help="the numbers of epochs for training of the global eigenbases")
parser.add_argument("--vgg", '-v', type=str, default='./pretrained_vgg_weights/vgg', 
                    help="the path to the pretrained VGG encoder")
parser.add_argument("--saveto", '-s', type=str, default='./saved_eigenbases/',
                    help="the path to the directory where to save the trained eigenbases")    

# It is not necessary to use MS-COCO and the image preprocessing specified in the file at './utils/DatasetAPI.py'.
parser.add_argument("--dataset", '-d', type=str, default='./MSCOCO/train2017', 
                    help="the path to the training dataset (MSCOCO)")
args = parser.parse_args()
    
    
n_ch = {0: cfg.new_channel0, 1: cfg.new_channel1, 2: cfg.new_channel2, 3: cfg.new_channel3}
vgg_encoder = VggEncoder(args.vgg)
vgg_sencoder = VggSEncoder()
ds = mscoco_dataset(args.dataset)

lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
manager = []
for b in range(4):
    ckpt = tf.train.Checkpoint(eigenbasis=vgg_sencoder.eigenbases[b])
    manager.append(tf.train.CheckpointManager(ckpt, f'{os.path.join(args.saveto, str(n_ch[b]))}', max_to_keep=1))

# warm up
train(vgg_encoder, vgg_sencoder, tf.random.normal([1, 256, 256, 3])) 
weights = [vgg_sencoder.eigenbases[0], vgg_sencoder.eigenbases[1], vgg_sencoder.eigenbases[2], vgg_sencoder.eigenbases[3]] 

for epoch in range(args.epochs):
    for i, imgs in enumerate(ds):
        with tf.GradientTape() as tape:
            loss = train(vgg_encoder, vgg_sencoder, imgs) 
            _loss = sum(loss)

        grads = tape.gradient(_loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        if (i+1) % 10 == 0:
            to_show = f"Epoch: {epoch+1}, iter: {i+1}, loss: {', '.join(map(lambda x: str(x.numpy()), loss))}"            
            print(to_show)                    
    
    for b in range(4):
        manager[b].save()


        
            