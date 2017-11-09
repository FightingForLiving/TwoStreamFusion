# coding=utf-8
import numpy as np
import os
import random
import csv
from scipy import misc

class ActionGenerator(object):
    def __init__(self, dataset_path,numClasses, nFramesPerVid, batch_size=8, validation_ratio=0.3):
        self.dataset_path = dataset_path
        self.img_path = os.path.join(dataset_path, 'jpegs')
        self.flow_path = os.path.join(dataset_path, 'flow')
        
        self.batch_size = batch_size
        self.numClasses = numClasses
        self.nFramesPerVid = nFramesPerVid
        
        self.train_list_avis = []
        self.test_list_avis = []
        self.val_list_avis = []
        self.avis_train_label={}

        with open(os.path.join(dataset_path, 'train_avis.txt')) as f:
            spam = csv.reader(f, delimiter=' ')
            for row in spam:
                self.train_list_avis.append(row[0].split('.')[0])
                self.avis_train_label[row[0].split('.')[0]] = row[1]

        # with open(os.path.join(dataset_path, 'val_avis.txt')) as f:
        #     spam = csv.reader(f, delimiter=' ')
        #     for row in spam:
        #         self.val_list_avis.append(row[0].split('.')[0])

        size_val_list = int(len(self.train_list_avis) * validation_ratio)
        self.val_list_avis = random.sample(self.train_list_avis, size_val_list)
        self.train_list_avis = list(filter(lambda x: x not in self.val_list_avis, self.train_list_avis))

        random.shuffle(self.train_list_avis)

    def generate_batch_list(self, avi_list):
        #avi_list = self.train_list_avis
        loop = 0
        batch_list = []
        max_size = len(avi_list)
        for _ in range(max_size//self.batch_size + 1):
            if loop + self.batch_size < max_size:
                gen_list = avi_list[loop:loop+self.batch_size]
            else:
                last_iter = loop + self.batch_size - max_size
                gen_list = avi_list[loop:max_size]
                loop = 0
                random.shuffle(self.train_list_avis)
            batch_list.append(gen_list)
            loop += self.batch_size
        return batch_list
    def train_batch_list(self):
        avi_list = self.train_list_avis
        return self.generate_batch_list(avi_list)

    def val_batch_list(self):
        avi_list = self.val_list_avis
        return self.generate_batch_list(avi_list)

    
    def get_images_flow(self, avi_name, height=224, width=224):#(self, img_path, flow_path, avi_name, height=224, width=224):
        """
        img_path: 'F:/actionRecognition/action_data_output'
        avi_name: 'neg(pos)_***'
        return nFramesPerVid rgb images with shape [nFramesPerVid, height, width, channels], channels=3
        and flow images with shape [nFramesPerVid, height, width, channels], channels=20
        """
        img_path = self.img_path
        flow_path = self.flow_path
    
        imgs = []
        flow = []
        imgs_output = []
        flow_output = []
        nStack = 20
        for img in os.listdir(os.path.join(img_path, avi_name)):
            imgs.append(img)
        nFrames = len(imgs)
        temporal_stride = np.random.randint(5,8) #产生[0,16)的随机整数
        img_index = [x for x in range(1,nFrames + 1)]
        #print("img_index:",img_index,avi_name)
        frame_samples = img_index[nStack//4:nFrames-nStack//4:temporal_stride]
        #print("frame_samples",frame_samples,avi_name)
        if len(frame_samples) >= self.nFramesPerVid:
            s = np.random.randint(len(frame_samples) - self.nFramesPerVid + 1)
            frame_samples = frame_samples[s:s+self.nFramesPerVid] #nFramesPerVid 张图片
        elif len(frame_samples) < self.nFramesPerVid:
            pass
        rgb_path = [os.path.join(img_path, avi_name, 'frame%06d.jpg' %x) for x in frame_samples]
        flows_path = []
        #print("frame_samples",frame_samples,avi_name)
        for x in frame_samples:
            for y in range(x-5, x+5):  #[x-5, x+5)  左闭右开
                flows_path.append(os.path.join(flow_path, 'u', avi_name, 'frame%06d.jpg' %y))
                flows_path.append(os.path.join(flow_path, 'v', avi_name, 'frame%06d.jpg' %y))
        for path in rgb_path: # 读取rgb图  5个
            img = misc.imread(path)
            img = misc.imresize(np.asarray(img), (height, width))
            imgs_output.append(img)
        imgs_output = np.asarray(imgs_output)

        for path in flows_path:  #读取光流图 100个
            img = misc.imread(path)
            img = misc.imresize(np.asarray(img), (height, width))
            flow_output.append(img)
        flow_output = np.asarray(flow_output)
        _, h, w = flow_output.shape
        flow_output = np.reshape(flow_output, [self.nFramesPerVid, nStack, h, w])  #[self.nFramesPerVid, nStack, h, w]
        return imgs_output, flow_output  #imgs_output: [5,224,224,3]  flow_output:[5,20,224,224]
    
    def get_batch(self, batch_list, height=224, width=224): #(self, batch_list, img_path, flow_path, height=224, width=224):  #batch_list: 一个batchsize的视频名字
        imgs_batch = np.zeros((2,2,2))  #可以使任意数字，任意维度
        flow_batch = np.zeros((2,2,2))
        for i, avi_name in enumerate(batch_list):
            imgsPerVid, flowPerVid = self.get_images_flow(avi_name, height, width)
            flowPerVid = flowPerVid.transpose((0,2,3,1))  #[5, 224, 224, 20]
            if i==0:
                imgs_batch = imgsPerVid
                flow_batch = flowPerVid
            else:
                imgs_batch = np.concatenate((imgs_batch, imgsPerVid), axis=0)
                flow_batch = np.concatenate((flow_batch, flowPerVid), axis=0)
        #labels = self.to_categorical([self.avis_train_label[x] for x in batch_list]) #labels.shape=[batch_size, numClasses]
        labels = [int(self.avis_train_label[x]) for x in batch_list]
                                    
        return imgs_batch, flow_batch, labels  #imgs_batch:[5*batch_size, 224, 224, 3]  flow_batch:[5*batch_size, 224, 224, 20]
