# This file is a part of StarNet code.
# https://github.com/nekitmm/starnet
# 
# StarNet is a neural network that can remove stars from images leaving only background.
# 
# The code attempts to support RGB (8bit) and single channel (8/16/32bit) input images.
# Output will be the same format as the input.
# 
# Copyright (c) 2018 Nikita Misiura
# http://www.astrobin.com/users/nekitmm/
# 
# This code is distributed on an "AS IS" BASIS WITHOUT WARRANTIES OF ANY KIND, express or implied.
# Please review LICENSE file before use.

import numpy as np
import tensorflow as tf
from PIL import Image as img
import matplotlib.pyplot as plt
import matplotlib
import sys
import time
import model
import starnet_utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

WINDOW_SIZE = 256                      # Size of the image fed to net. Do not change until you know what you are doing! Default is 256
                                       # and changing this will force you to train the net anew.

def transform(image, stride):
    
    # placeholders for tensorflow
    X = tf.compat.v1.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "X")
    Y = tf.compat.v1.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "Y")

    # create model
    train, avers, outputs = model.model(X, Y)
    
    #initialize variables
    init = tf.compat.v1.global_variables_initializer()
    
    # create saver instance to load model parameters
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:        
        # initialize all variables and start training
        sess.run(init)
        
        # restore current state of the model
        print("Restoring previous state of the model...")
        saver.restore(sess, "./model.ckpt")
        print("Done!")
        
        # read input image
        
        print("Opening input image...")
        input_image = img.open(image)
        input = np.array(input_image, dtype = np.float32)
        
        pixel_type = np.uint8
        if ";16" in input_image.mode:
            pixel_type = np.uint16
        elif input_image.mode == "I":
            pixel_type = np.uint32
        elif input_image.mode == "F":
            pixel_type = np.float32
        
        # rescale to [0 1] based on image bit depth
        outputscale = None
        if pixel_type is not np.float32:
            outputscale = 2**(pixel_type().itemsize * 8) - 1
            input /= outputscale
        
        # tf expects rgb images
        mono = len(input.shape) == 2
        if mono :
            input = np.dstack( ( input, input, input ) )
        
        print("Done!")
        
        # backup to use for mask
        backup = np.copy(input)
        
        # rescale to [-1, 1]
        input = input * 2 - 1
        
        # now some tricky magic
        # image size is unlikely to be multiple of stride and hence we need to pad the image and
        # also we need some additional padding to allow offsets on sides of the image
        offset = int((WINDOW_SIZE - stride) / 2)
        
        # get size of the image and calculate numbers of iterations needed to transform it
        # given stride and taking into account that we will pad it a bit later (+1 comes from that)
        h, w, _ = input.shape
        ith = int(h / stride) + 1
        itw = int(w / stride) + 1
        
        # calculate how much we need to add to make image sizes multiples of stride
        dh = ith * stride - h
        dw = itw * stride - w
        
        # pad image using parts of the image itself and values calculated above
        input = np.concatenate((input, input[(h - dh) :, :, :]), axis = 0)
        input = np.concatenate((input, input[:, (w - dw) :, :]), axis = 1)
        
        # get image size again and pad to allow offsets on all four sides of the image
        h, w, _ = input.shape
        input = np.concatenate((input, input[(h - offset) :, :, :]), axis = 0)
        input = np.concatenate((input[: offset, :, :], input), axis = 0)
        input = np.concatenate((input, input[:, (w - offset) :, :]), axis = 1)
        input = np.concatenate((input[:, : offset, :], input), axis = 1)
        
        # copy input image to output
        output = np.copy(input)
        
        # helper array just to add fourth dimension to net input
        tmp = np.zeros((1, WINDOW_SIZE, WINDOW_SIZE, 3), dtype = np.float)
        
        # here goes
        for i in range(ith):
            for j in range(itw):
                print('Transforming input image... %d%%\r' % int((itw * i + j + 1) * 100 / (ith * itw)))
                
                x = stride * i
                y = stride * j
                
                # write piece of input image to tmp array
                tmp[0] = input[x : x + WINDOW_SIZE, y : y + WINDOW_SIZE, :]
                
                # transform
                result = sess.run(outputs, feed_dict = {X:tmp})
                
                # write transformed array to output
                output[x + offset : x + stride + offset, y + offset: y + stride + offset, :] = result[0, offset : stride + offset, offset : stride + offset, :]
        print("Transforming input image... Done!")
        
        # rescale back to [0, 1]
        output = (output + 1) / 2

        output = output.clip(0, None)
        if pixel_type is not np.float32 :
            output = output.clip(None, 1)
        
        # leave only necessary part, without pads added earlier
        output = output[offset : - (offset + dh), offset : - (offset + dw), :]
        
        print("Saving mask...")
        # mask showing areas that were changed significantly
        mask = (((backup * 255).astype(np.uint8) - (output * 255).astype(np.uint8)) > 25).astype(np.uint8)
        mask = mask.max(axis = 2, keepdims = True)
        mask = np.concatenate((mask, mask, mask), axis = 2)
        img.fromarray(mask * 255).save('./' + image + '_mask.tif')
        print("Done!")

        print("Saving output image [%dbit %s]..." % ( (pixel_type().itemsize * 8), "Y" if mono else "RGB" ) )
        if mono:
            output = output[:,:,0]
        if outputscale is not None:
            output *= outputscale
        img.fromarray( output.astype(pixel_type), mode=input_image.mode ).save('./' + image + '_starless.tif')
        print("Done!")
