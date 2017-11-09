# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
from dataset_generator import ActionGenerator

class vgg16(object):
    def __init__(self, img_inputs, flow_inputs, nClasses, nFramesPerVid, weight_file, weights=None, sess=None):
        self.nClasses = nClasses
        self.img_inputs = img_inputs
        self.flow_inputs = flow_inputs
        self.nFramesPerVid = nFramesPerVid
        self.weight_file = weight_file
        #self.model()


    def model(self):
        ##################################################################### vgg16-spatial ############################################################################
        with tf.variable_scope('conv1_1_spatial'):   # imgs_inputs [?, 224, 224, 3]
            weights = tf.get_variable("W", [3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.img_inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv1_1_spatial = tf.nn.relu(conv + biases)    #[?,224,224,64]
                                               
        with tf.variable_scope('conv1_2_spatial'):
            weights = tf.get_variable("W", [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv1_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv1_2_spatial = tf.nn.relu(conv + biases) #[?,224,224,64]
            
        with tf.variable_scope('pool1_spatial'):
            pool1_spatial = tf.nn.max_pool(conv1_2_spatial, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_spatial') #[?, 112,112,64]

            
            
        with tf.variable_scope('conv2_1_spatial'):
            weights = tf.get_variable("W", [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2_1_spatial = tf.nn.relu(conv + biases) #[?,112,112,128]
            
        with tf.variable_scope('conv2_2_spatial'):
            weights = tf.get_variable("W", [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv2_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2_2_spatial = tf.nn.relu(conv + biases) #[?,112,112,128]
            
        with tf.variable_scope('pool2_spatial'):
            pool2_spatial = tf.nn.max_pool(conv2_2_spatial, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_spatial') #[?, 56,56,128]



        with tf.variable_scope('conv3_1_spatial'):
            weights = tf.get_variable("W", [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_1_spatial = tf.nn.relu(conv + biases)#[?,56,56,256]
            
        with tf.variable_scope('conv3_2_spatial'):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv3_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_2_spatial = tf.nn.relu(conv + biases) #[?,56,56,256]
            
        with tf.variable_scope('conv3_3_spatial'):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv3_2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_3_spatial = tf.nn.relu(conv + biases) #[?,56,56,256]
            
        with tf.variable_scope('pool3_spatial'):
            pool3_spatial = tf.nn.max_pool(conv3_3_spatial, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_spatial') #[?, 28,28,256]



        with tf.variable_scope('conv4_1_spatial'):
            weights = tf.get_variable("W", [3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool3_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_1_spatial = tf.nn.relu(conv + biases) #[?,28,28,512]
            
        with tf.variable_scope('conv4_2_spatial'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv4_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_2_spatial = tf.nn.relu(conv + biases) #[?,28,28,512]
            
        with tf.variable_scope('conv4_3_spatial'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv4_2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_3_spatial = tf.nn.relu(conv + biases) #[?,28,28,512]
            
        with tf.variable_scope('pool4_spatial'):
            pool4_spatial = tf.nn.max_pool(conv4_3_spatial, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4_spatial') #[?, 14,14,512]

            

        with tf.variable_scope('conv5_1_spatial'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool4_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_1_spatial = tf.nn.relu(conv + biases) #[?,14,14,512]
            
        with tf.variable_scope('conv5_2_spatial'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv5_1_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_2_spatial = tf.nn.relu(conv + biases) #[?,14,14,512]
            
        with tf.variable_scope('conv5_3_spatial'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv5_2_spatial, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_3_spatial = tf.nn.relu(conv + biases) #[?,14,14,512]
            
        #with tf.variable_scope('pool5_spatial'):
            #pool5_spatial = tf.layers.max_pooling2d(conv5_3_spatial, pool_size=[2,2], strides=(2,2), padding='valid') #[?, 7,7,512]

        

        ##################################################################### vgg16-temporal ############################################################################
        with tf.variable_scope('conv1_1_temporal'): # flow_imputs: [batch_size*nFramesPerVid, height, width, nStacks]  nStacks=20
            weights = tf.get_variable("W", [3,3,20,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.flow_inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv1_1_temporal = tf.nn.relu(conv + biases)    #[?,224,224,64]
                                               
        with tf.variable_scope('conv1_2_temporal'):
            weights = tf.get_variable("W", [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv1_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv1_2_temporal = tf.nn.relu(conv + biases) #[?,224,224,64]
            
        with tf.variable_scope('pool1_temporal'):
            pool1_temporal = tf.nn.max_pool(conv1_2_temporal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_temporal') #[?, 112,112,64]

            
            
        with tf.variable_scope('conv2_1_temporal'):
            weights = tf.get_variable("W", [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2_1_temporal = tf.nn.relu(conv + biases) #[?,112,112,128]
            
        with tf.variable_scope('conv2_2_temporal'):
            weights = tf.get_variable("W", [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv2_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv2_2_temporal = tf.nn.relu(conv + biases) #[?,112,112,128]
            
        with tf.variable_scope('pool2_temporal'):
            pool2_temporal = tf.nn.max_pool(conv2_2_temporal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_temporal') #[?, 56,56,128]



        with tf.variable_scope('conv3_1_temporal'):
            weights = tf.get_variable("W", [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_1_temporal = tf.nn.relu(conv + biases)#[?,56,56,256]
            
        with tf.variable_scope('conv3_2_temporal'):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv3_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_2_temporal = tf.nn.relu(conv + biases) #[?,56,56,256]
            
        with tf.variable_scope('conv3_3_temporal'):
            weights = tf.get_variable("W", [3,3,256,256], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv3_2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv3_3_temporal = tf.nn.relu(conv + biases) #[?,56,56,256]
            
        with tf.variable_scope('pool3_temporal'):
            pool3_temporal = tf.nn.max_pool(conv3_3_temporal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_temporal') #[?, 28,28,256]



        with tf.variable_scope('conv4_1_temporal'):
            weights = tf.get_variable("W", [3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool3_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_1_temporal = tf.nn.relu(conv + biases) #[?,28,28,512]
            
        with tf.variable_scope('conv4_2_temporal'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv4_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_2_temporal = tf.nn.relu(conv + biases) #[?,28,28,512]
            
        with tf.variable_scope('conv4_3_temporal'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv4_2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv4_3_temporal = tf.nn.relu(conv + biases) #[?,28,28,512]
            
        with tf.variable_scope('pool4_temporal'):
            pool4_temporal = tf.nn.max_pool(conv4_3_temporal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4_temporal') #[?, 14,14,512]

            

        with tf.variable_scope('conv5_1_temporal'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(pool4_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_1_temporal = tf.nn.relu(conv + biases) #[?,14,14,512]
            
        with tf.variable_scope('conv5_2_temporal'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv5_1_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_2_temporal = tf.nn.relu(conv + biases) #[?,14,14,512]
            
        with tf.variable_scope('conv5_3_temporal'):
            weights = tf.get_variable("W", [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(conv5_2_temporal, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv5_3_temporal = tf.nn.relu(conv + biases) #[?,14,14,512]   此处往上?=batchsize*nFramesPerVid

        #     conv5_3_reshape = tf.reshape(conv5_3_temporal,[-1,self.nFramesPerVid,14,14,512]) #[?,self.nFramesPerVid, 14, 14, 512]
        # with tf.variable_scope('pool5_temporal'):
        #     pool5_temporal = tf.layers.max_pooling3d(conv5_3_reshape, pool_size=[self.nFramesPerVid,2,2], strides=(2,2,2), padding='valid') #[?,1,7,7,512]



        # with tf.variable_scope('fc6_temporal'):
        #     pool5_temporal_flat = tf.reshape(pool5_temporal,[-1, 7*7*512])
        #     fc6_W = tf.get_variable('W', [7*7*512, 4096], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        #     fc6_b = tf.get_variable('b',[4096], initializer=tf.constant_initializer(0.1), trainable=True)
        #     temporal_fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool5_temporal_flat, fc6_W), fc6_b))
        # with tf.variable_scope('fc7_temporal'):
        #     fc7_W = tf.get_variable('W', [4096, 4096], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        #     fc7_b = tf.get_variable('b',[4096], initializer=tf.constant_initializer(0.1), trainable=True)
        #     temporal_fc7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(temporal_fc6, fc7_W), fc7_b))
        # with tf.variable_scope('fc8_temporal'):
        #     fc8_W = tf.get_variable('W', [4096, self.nClasses], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        #     fc8_b = tf.get_variable('b',[self.nClasses], initializer=tf.constant_initializer(0.1), trainable=True)
        #     temporal_fc8 = tf.nn.bias_add(tf.matmul(temporal_fc7, fc8_W), fc8_b)
        


        ####将conv5_3_spatial 和conv5_3_temporal 拼接起来，[?, 14, 14, 1024]  spatial_temporal_concat
        ################################################################## spatial_temporal_fusion ################################################################################
        spatial_temporal_concat = tf.concat([conv5_3_spatial, conv5_3_temporal],3) #np.concatenate((conv5_3_spatial, conv5_3_temporal), axis=3) # [?, 14, 14, 1024]
        fusion_reshape = tf.reshape(spatial_temporal_concat,[-1, self.nFramesPerVid, 14, 14, 1024])   #[?, self.nFramesPerVid, 14, 14, 1024]
        
        fusion_conv6 = tf.layers.conv3d(fusion_reshape, filters=512, kernel_size=[3,3,3], strides=(1,1,1), padding='same', activation=tf.nn.relu) #[?,self.nFramesPerVid,14,14,512]
        pool3d = tf.layers.max_pooling3d(fusion_conv6, pool_size=[self.nFramesPerVid,2,2], strides=(2,2,2), padding='valid') # [?,1,7,7,512]  ?=batchsize 
        pool3d_flat = tf.reshape(pool3d, [-1, 7*7*512])
        with tf.variable_scope('fc6_spatial'):
            fc6_W = tf.get_variable('W', [7*7*512, 4096], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            fc6_b = tf.get_variable('b',[4096], initializer=tf.constant_initializer(0.1), trainable=True)
            fusion_fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool3d_flat, fc6_W), fc6_b))
        with tf.variable_scope('fc7_spatial'):
            fc7_W = tf.get_variable('W', [4096, 4096], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            fc7_b = tf.get_variable('b',[4096], initializer=tf.constant_initializer(0.1), trainable=True)
            fusion_fc7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fusion_fc6, fc7_W), fc7_b))
        with tf.variable_scope('fc8_spatial'):
            fc8_W = tf.get_variable('W', [4096, self.nClasses], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            fc8_b = tf.get_variable('b',[self.nClasses], initializer=tf.constant_initializer(0.1), trainable=True)
            fusion_fc8 = tf.nn.bias_add(tf.matmul(fusion_fc7, fc8_W), fc8_b)

        return fusion_fc8#, temporal_fc8

    def load_initial_weights(self, session):
        weights_dict = np.load(self.weight_file, encoding = 'bytes')
        vgg_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'] #, 'fc6', 'fc7']
        print("*********####################################################################******")
        for layer in vgg_layers:
            if layer != 'conv1_1':
                with tf.variable_scope(layer+"_spatial", reuse = True):
                    var = tf.get_variable('W', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_W']))
                    var = tf.get_variable('b', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_b']))
                with tf.variable_scope(layer+"_temporal", reuse = True):
                    var = tf.get_variable('W', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_W']))
                    var = tf.get_variable('b', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_b']))
            else:
                with tf.variable_scope(layer+"_spatial", reuse = True):
                    var = tf.get_variable('W', trainable = True)
                    session.run(var.assign(weights_dict[layer+'_W']))
                    var = tf.get_variable('b', trainable = True)
                session.run(var.assign(weights_dict[layer+'_b']))
        vgg_fc_layers = ['fc6', 'fc7']
        for layer in vgg_fc_layers:
            with tf.variable_scope(layer+'_spatial', reuse = True):
                var = tf.get_variable('W', trainable=True)
                session.run(var.assign(weights_dict[layer+'_W']))
            with tf.variable_scope(layer+'_spatial', reuse = True):
                var = tf.get_variable('b', trainable=True)
                session.run(var.assign(weights_dict[layer+'_b']))


if __name__ == '__main__':
    #设置gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    nClasses = 2
    nStacks = 20
    height = 224
    width = 224
    nFramesPerVid = 2
    batch_size = 8
    training_epoch = 500

    imgs = tf.placeholder(tf.float32, [None, height, width, 3])
    flows = tf.placeholder(tf.float32, [None, height, width, nStacks])
    target = tf.placeholder("int64", [None])

    weight_file = '/home/ubuntu/My_fusion/model/vgg16_weights.npz'

    vgg = vgg16(imgs, flows, nClasses, nFramesPerVid, weight_file)

    fusion_yout = vgg.model() #########   
    learning_rate = 1e-3
    print("fusion_yout.shape:",fusion_yout.shape)
    fusion_total_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(target, nClasses), logits = fusion_yout)
    fusion_mean_loss = tf.reduce_mean(fusion_total_loss)    
    fusion_optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay=0.9, momentum = 0.9)
    fusion_train_step = fusion_optimizer.minimize(fusion_mean_loss)


    fusion_prediction = tf.equal(tf.argmax(fusion_yout, 1), tf.argmax(tf.one_hot(target, nClasses), 1))
    fusion_accuracy = tf.reduce_mean(tf.cast(fusion_prediction, tf.float32))

    # temporal_total_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(target, nClasses), logits = temporal_yout)
    # temporal_mean_loss = tf.reduce_mean(temporal_total_loss)
    # temporal_optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate, decay=0.9, momentum=0.9)
    # temporal_train_step = temporal_optimizer.minimize(temporal_mean_loss)


    # temporal_prediction = tf.equal(tf.argmax(temporal_yout, 1), tf.argmax(tf.one_hot(target, nClasses), 1))
    # temporal_accuracy = tf.reduce_mean(tf.cast(temporal_prediction, tf.float32))




    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    vgg.load_initial_weights(sess)    #加载vgg16预训练模型
    dataset_path = '/home/ubuntu/My_fusion/data'
    checkpoint_path_finetune = '/home/ubuntu/My_fusion/model/' #'/home/ubuntu/My_fusion/model/'


    #######################################训练一半停止时，加载之前的模型，继续训练#######################
    #saver.restore(sess, '/home/ubuntu/My_fusion/model/modelfinetune.ckpt-301')
    #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path_finetune))#(os.path.join(checkpoint_path_finetune,'modelfinetune.ckpt')))

    validation_ratio = 0.3
    obj = ActionGenerator(dataset_path, nClasses, nFramesPerVid, batch_size, validation_ratio)
    for epoch in range(training_epoch+1):
        batch_list = obj.train_batch_list()
        
        fusion_avg = 0
        fusion_cost = 0
        temporal_avg = 0
        temporal_cost = 0

        total_batch = len(batch_list)
        
        for i, batch in enumerate(batch_list):
            input_imgs, input_flow, labels = obj.get_batch(batch, height, width)
            start = time.time()
            #labels = tf.one_hot(target, nClasses)

            sess.run(fusion_train_step, feed_dict={imgs:input_imgs, flows:input_flow, target:labels})
            #sess.run(temporal_train_step, feed_dict={imgs: input_imgs, flows:input_flow, target:labels})
            fu_cost = sess.run(fusion_mean_loss, feed_dict={imgs: input_imgs, flows:input_flow, target:labels})
            fu_acc = fusion_accuracy.eval(feed_dict={imgs: input_imgs, flows:input_flow, target:labels}, session=sess)
            # tp_cost = sess.run(temporal_mean_loss, feed_dict={imgs: input_imgs, flows:input_flow, target:labels})
            # tp_acc = temporal_accuracy.eval(feed_dict={imgs: input_imgs, flows:input_flow, target:labels}, session=sess)


            fusion_avg += fu_acc
            fusion_cost += fu_cost
            # temporal_avg += tp_acc
            # temporal_cost += tp_cost
            # if i % 20 == 0:
            #     print("Epoch:%03d" %(epoch+1)," Step: %03d" %i, "time to run 20 batches:", time.time() - start, "seconds", "fusion loss:",fusion_cost/total_batch,"temporal loss:" temporal_cost/total_batch)
        print("========================================================== epoch"+str(epoch)+" ========================================================")
        print("Fusion train accuracy:", fusion_avg/total_batch, " Fusion loss:",fusion_cost/total_batch)
        #print("Temporal train accuracy:", temporal_avg/total_batch, " Temporal loss:",temporal_cost/total_batch)

        ################################################################ validation  ##################################################################
        if epoch % 5 == 0:
            batch_list = obj.val_batch_list()
            fusion_avg = 0
            fusion_cost = 0
            temporal_avg = 0
            temporal_cost = 0
            true_positives = 0

            total_batch = len(batch_list)            
            for i, batch in enumerate(batch_list):
                input_imgs, input_flow, labels = obj.get_batch(batch, height, width)
                #labels = tf.one_hot(target, nClasses)
                fu_cost = sess.run(fusion_mean_loss, feed_dict={imgs: input_imgs, flows:input_flow, target:labels})
                fu_acc = fusion_accuracy.eval(feed_dict={imgs: input_imgs, flows:input_flow, target:labels}, session=sess)
                # tp_cost = sess.run(temporal_mean_loss, feed_dict={imgs: input_imgs, flows:input_flow, target:labels})
                # tp_acc = temporal_accuracy.eval(feed_dict={imgs: input_imgs, flows:input_flow, target:labels}, session=sess)

                fusion_avg += fu_acc
                fusion_cost += fu_cost
                # temporal_avg += tp_acc
                # temporal_cost += tp_cost
            print("Fusion validation accuracy:", fusion_avg/total_batch, " Fusion loss:",fusion_cost/total_batch)
            #print("Temporal validation accuracy:", temporal_avg/total_batch, " Temporal loss:",temporal_cost/total_batch)

        if epoch % 20 == 0:    
            checkpoint_path = os.path.join(checkpoint_path_finetune, 'modelfinetune.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch+1)
            print("saved to " + checkpoint_path)

