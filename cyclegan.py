#!/usr/bin/env python
# coding: utf-8

# In[4]:


#必要なモジュールをインポート
from __future__ import print_function, division
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import save_model
from data_loader import DataLoader
import scipy
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import imageio


# In[5]:


class CycleGAN():
    
    #ニューラルネットワーク構築
    def __init__(self):
        # 入力画像サイズ
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # 画像インポート
        self.dataset_name = 'oil painting'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # PatchGANの出力設定
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # genaratorの最初の層のフィルタ（重み）数
        self.gf = 32
        
        # discriminatorの最初の層のフィルタ（重み）数
        self.df = 64

        # サイクル一貫性損失の重み
        self.lambda_cycle = 10.0                    
        
        #同一性損失の重み
        self.lambda_id = 0.9 * self.lambda_cycle   

        optimizer = Adam(0.0002, 0.5)
        
        # 識別器の構成とコンパイル
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])


        # 構成とコンパイル
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # ドメイン画像の入力
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        #画像を翻訳
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
       
        # 画像を再翻訳
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        
        # 画像の恒等写真
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # 複合モデルに対して、生成器のみを訓練
        self.d_A.trainable = False
        self.d_B.trainable = False

        # 翻訳した画像の妥当性を評価する識別器
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # 生成器の訓練
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    #U-NET構造の生成器の生成
    def build_generator(self):
        
        #生成器の層
        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

       #アップサンプリング用の層
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # 画像入力
        d0 = Input(shape=self.img_shape)

        # ダウンサンプリング
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # アップサンプリング
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    #識別器の生成
    def build_discriminator(self):

        #識別器の層
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    #CyclGANの訓練
    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # 敵対性損失の正解ラベル
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i,imgs_A,imgs_B in self.data_loader.load_batch(batch_size):
                
                
                #識別器の訓練
                # 画像A→Bへの翻訳
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # 元画像(real)、翻訳された画像(fake)の訓練
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # 識別器の損失
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


            
                #生成器の訓練
                # 生成器の損失
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                #学習時間
                elapsed_time = datetime.datetime.now() - start_time

                # 学習進捗表示
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "                                                                         % ( epoch+1, epochs,
                                                                            batch_i, (self.data_loader.n_batches-2),
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # 画像サンプルの保存
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
            
            if epoch < epochs-1 :  
                self.g_BA.save_weights(f'./weights/backup/oil_painting_weights_{epoch+1}.h5')
            
            else :
                self.g_BA.save_weights('./weights/oil_painting_weights.h5')

    #画像サンプルの保存設定
    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # 試験用
        #imgs_A = self.data_loader.load_img('datasets/oil painting/testA/00020.jpg')
        #imgs_B = self.data_loader.load_img('datasets/oil painting/testB/2014-12-31 16_09_24.jpg')

        # 画像の翻訳
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        
        # 画像の再翻訳
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # 画像のリスケール
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()
          


# In[ ]:


if __name__ == '__main__':
    gan = CycleGAN()   
    gan.train(epochs=200, batch_size=20,sample_interval=50)


# In[ ]:




