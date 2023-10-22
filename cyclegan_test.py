#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import scipy
from glob import glob
from imageio import imread
from PIL import Image


# In[11]:


class CycleGAN():
    
    #ニューラルネットワーク構築
    def __init__(self):
        # 入力画像サイズ
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

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
        #識別機
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


# In[13]:
gan = CycleGAN()
gan.g_BA.load_weights("C:/Users/tasuk/OneDrive/デスクトップ/ポートフォリオ_Samurai/cyclegan/weights/oil_painting_weights.h5")
path = glob("C:/Users/tasuk/OneDrive/デスクトップ/ポートフォリオ_Samurai/cyclegan/original_image/*")
img_name = os.path.splitext(os.path.basename(path[0]))[0]
img = imread(path[0], pilmode="RGB")
img_res=(128, 128)
img = np.array(Image.fromarray(img.astype(np.uint8)).resize(img_res))
img = img/127.5 - 1.
img=img[np.newaxis, :, :, :]
result = gan.g_BA.predict(img)
result_image = Image.fromarray(((result[0] + 1) * 127.5).astype(np.uint8))
result_image.save(os.path.join("C:/Users/tasuk/OneDrive/デスクトップ/ポートフォリオ_Samurai/cyclegan/result/", f'{img_name}_fake.png'))




