import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from utils.vgg_model import VggEncoder
from utils.lightweight_model import VggSDecoder, VggSEncoder
from utils.DatasetAPI import mscoco_dataset
from utils.config import cfg
import os, sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

mse = tf.keras.losses.MeanSquaredError()
def train(vgg_encoder, vgg_sencoder, vgg_sdecoder, block, input_img):
    enc_feats = []
    senc_feats = []
    skip_feats = []

    x = input_img
    for l in range(block+1):
        x = vgg_encoder(l, x)
        enc_feats.append(x)

    x = input_img
    for l in range(block+1):
        x = vgg_sencoder(l, x)
        senc_feats.append(x[0])
        skip_feats.append(x[1])
        x = x[0]

    m1 = tf.reduce_mean(enc_feats[-1], axis=[1,2], keepdims=True)
    m2 = tf.reduce_mean(senc_feats[-1], axis=[1,2], keepdims=True)
    loss_distill = mse(tf.linalg.einsum('bijd,cd->bijc', senc_feats[-1]-m2, vgg_sencoder.eigenbases[block]), enc_feats[-1]-m1)

    sdec_feats = []    
    for l in reversed(range(block+1)):
        x = vgg_sdecoder(l, x, skip=skip_feats[l])
        sdec_feats.append(x)

    loss_rec = mse(input_img, x)

    if block == 0:
        return loss_distill, loss_rec

    for l in range(block+1):
        x = vgg_encoder(l, x)            

    loss_percept = mse(enc_feats[-1], x)
    loss_feat_rec = mse(senc_feats[-2], sdec_feats[0])

    return loss_distill, loss_rec, loss_feat_rec, loss_percept




parser = argparse.ArgumentParser()

''' 
The VGG encoder is splitted into a series of blocks (enc0, enc1, enc2, enc3) from the input to the bottleneck.
The lightweight encoder is splitted into blocks (senc0, senc1, senc2, senc3) from the input to the bottleneck.
The lightweight decoder is splitted into (sdec3, sdec2, sdec1, sdec0) from the bottlenect to the output. sencN and sdecN form a block pair. 
PCA knowledge distillation trains the four block pairs from N=0 to N=3 by distilling information from encN.
However, sometimes the training of a certain pair, say N=2, may fail. 
We want to restart the training from N=2 with the former pairs restored from the saved checkpoints.
'''
parser.add_argument('--pair', '-p', type=int, choices=[0, 1, 2, 3], default=0, 
                    help="which pair of a lightweight encoder block and a lightweight decoder block to begin training from")
'''
By default, the numbers of epochs for the training of the four pairs are 15, 10, 10, and 10 for N=0, 1, 2, and 3, respectively.
If training does not start from N=0, say N=2, then the first two arguments 15 and 10 will have no effects.
'''
parser.add_argument('--epochs', '-i', type=int, nargs=4, default=[15, 10, 10, 10],
                    help="the numbers of epochs for training of the four block pairs")


parser.add_argument("--vgg", '-v', type=str, default='./pretrained_vgg_weights/vgg', 
                    help="the path to the pretrained VGG encoder")
parser.add_argument("--eigenbases", '-e', type=str, default='./saved_eigenbases/', 
                    help="the path to the trained eigenbases from train_eigenbases.py")
parser.add_argument("--saveto", '-s', type=str, default='./saved_lightweight_model/',
                    help="the path to the directory where to save the trained lightweight model")    

# It is not necessary to use MS-COCO and the image preprocessing specified in the file at './utils/DatasetAPI.py'.
parser.add_argument("--dataset", '-d', type=str, default='./MSCOCO/train2017', 
                    help="the path to the training dataset (MSCOCO)")

args = parser.parse_args()



vgg_encoder = VggEncoder(args.vgg)
vgg_sencoder = VggSEncoder()
vgg_sdecoder = VggSDecoder()

blocks = range(args.pair, 4)
ds = mscoco_dataset(args.dataset)

if args.pair != 0:
    for i in range(0, args.pair):
        ckpt = tf.train.Checkpoint(senc=vgg_sencoder.btnecks[i], sdec=vgg_sdecoder.btnecks[i])
        try:
            ckpt.restore(tf.train.latest_checkpoint(os.path.join(args.saveto, str(i), 'model'))).assert_consumed()
        except:
            print(f"The {i}-th block pair have to be trained before training the block pair {args.pair}.")
            quit()

n_ch = {0: cfg.new_channel0, 1: cfg.new_channel1, 2: cfg.new_channel2, 3: cfg.new_channel3}
for b in range(4):
    ckpt = tf.train.Checkpoint(eigenbasis=vgg_sencoder.eigenbases[b])
    try:
        ckpt.restore(tf.train.latest_checkpoint(f'{os.path.join(args.eigenbases, str(n_ch[b]))}')).assert_consumed()
    except:
        print(f"Please train eigenbases before distillation.")
        quit()


for block in blocks:
    lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt_opt = tf.train.Checkpoint(optimizer=optimizer)
    ckpt_aec = tf.train.Checkpoint(senc=vgg_sencoder.btnecks[block], sdec=vgg_sdecoder.btnecks[block])
    manager_opt = tf.train.CheckpointManager(ckpt_opt, f"{os.path.join(args.saveto, str(block), 'opt')}", max_to_keep=1) 
    manager_aec = tf.train.CheckpointManager(ckpt_aec, f"{os.path.join(args.saveto, str(block), 'model')}", max_to_keep=1) 
    
    # warm up
    train(vgg_encoder, vgg_sencoder, vgg_sdecoder, block, tf.random.normal([1, 256, 256, 3]))
    weights = vgg_sencoder.btnecks[block].trainable_weights + vgg_sdecoder.btnecks[block].trainable_weights 

    for epoch in range(args.epochs[block]):
        for i, imgs in enumerate(ds):
            with tf.GradientTape() as tape:
                loss = train(vgg_encoder, vgg_sencoder, vgg_sdecoder, block, imgs) 
                _loss = sum(loss)

            grads = tape.gradient(_loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            if (i+1) % 10 == 0:
                to_show = f"Epoch: {epoch+1}, iter: {i+1}, loss: {', '.join(map(lambda x: str(x.numpy()), loss))}"
                print(to_show)                    

        manager_opt.save()
        manager_aec.save()
    
    if block == 3:
        vgg_sencoder.save_weights(f"{os.path.join(args.saveto, 'sencoder')}")
        vgg_sdecoder.save_weights(f"{os.path.join(args.saveto, 'sdecoder')}")

        
            