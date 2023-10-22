#!/usr/bin/env python
# coding: utf-8

# In[31]:


import scipy
from glob import glob
from imageio import imread
import numpy as np
from PIL import Image


# In[33]:


class DataLoader():
    
    #データセットの名前と画像の解像度の初期化
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        
    #入力された画像の前処理(リサイズ、正規化)
    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)
        

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = np.array(Image.fromarray(img.astype(np.uint8)).resize(self.img_res))

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = np.array(Image.fromarray(img.astype(np.uint8)).resize(self.img_res))
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    #バッチごとにimgs_A, imgs_Bを生成
    def load_batch(self, batch_size=1, is_testing=False, current_batch=0):
        data_type = "train" if not is_testing else "test"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = np.array(Image.fromarray(img_A.astype(np.uint8)).resize(self.img_res))
                img_B = np.array(Image.fromarray(img_B.astype(np.uint8)).resize(self.img_res))

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield current_batch + i,imgs_A,imgs_B

    #試験用画像読み込みメソッドの作成
    def load_img(self, path):
        img = self.imread(path)
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize(self.img_res))
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]
    
    #imreadメソッドの再定義
    def imread(self, path):
        return imread(path, pilmode="RGB").astype(np.float)


# In[ ]:




