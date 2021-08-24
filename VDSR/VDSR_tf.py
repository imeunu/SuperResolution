import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

imgs = []
sub_img_size = 41
ch_num = 3

dir = '/Users/kimminsu/Desktop/HYU/2021 after army/DeepLearning/train'

for i in range(1,92):
    img_path = dir + "/" + str(i)
    try:img = plt.imread(img_path +'.bmp')
    except:img = plt.imread(img_path + '.jpg')
    img = tf.image.extract_patches(np.array([img]),
                                   sizes = [1,sub_img_size,sub_img_size,1],
                                   strides = [1,sub_img_size,sub_img_size,1],
                                   rates = [1,1,1,1],padding = 'VALID')
    img = tf.reshape(img, [-1,sub_img_size,sub_img_size,3])
    imgs.append(img)
imgs = np.vstack(imgs)

tr_size = len(imgs)//10*8
val_size = imgs[0] - tr_size

np.random.shuffle(imgs)

tr_set = imgs[:tr_size]
val_set = imgs[tr_size:]


class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2)])
        
        self.cnn_in = tf.keras.layers.Conv2D(
            input_shape=(None,None,ch_num),
            kernel_size = 3,filters = 64,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.cnns=tf.keras.Sequential()
        for i in range(18):
            self.cnns.add(tf.keras.layers.Conv2D(
                kernel_size = 3,filters = 64,
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        self.cnn_out = tf.keras.layers.Conv2D(
            kernel_size = 3,filters = ch_num,
            padding='same',
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(0.001))
    def call(self,imgs,scale = [2,3,4],downSample=False,training=False):
        if training:
            l = [imgs]
            for i in range(3): l.append(self.augmentation(imgs))
            imgs = tf.stack(l,axis=0)
            imgs = tf.reshape(imgs,[-1,imgs.shape[2],imgs.shape[3],ch_num])

        imgs = tf.cast(imgs,dtype=tf.float32)
        imgs = (imgs-127.5)/255
        rand_s = np.random.choice(scale)

        if not downSample:
            X = tf.image.resize(
                imgs,[sub_img_size*rand_s,sub_img_size*rand_s],
                method='bicubic')
            X = tf.image.resize(
                imgs,
                [sub_img_size,sub_img_size],method='bicubic')
        if downSample:
            L=int(sub_img_size/rand_s)
            X = tf.image.resize(imgs,[L,L],
                                method='bicubic',
                                antialias = True)
            X = tf.image.resize(imgs,[sub_img_size,sub_img_size],
                                method='bicubic')

        residual = self.cnn_in(X)
        redidual = self.cnns(residual)
        residual = self.cnn_out(residual)

        output = tf.keras.layers.Add()((residual,X))
        if not downSample: return output
        loss = tf.reduce_mean(tf.square(imgs-output))
        bicubic_loss = tf.reduce_mean(tf.square(imgs-X))
        return output, loss, bicubic_loss
model = Model()

    
epoch = 50
batch_size = 16
tr_tensor = tf.data.Dataset.from_tensor_slices(tr_set).batch(batch_size)
val_tensor = tf.data.Dataset.from_tensor_slices(val_set).batch(batch_size)
tr_num = tr_set.shape[0]//batch_size + 1
val_num = val_set.shape[0]//batch_size + 1
lrning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.001,
                                                             tr_num*10,
                                                             0.5,
                                                             staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lrning_rate)
tr_loss_list = []
val_loss_list = []
bicubic_tr_loss_list = []
bicubic_val_loss_list = []
last_epoch = 0

for e in range(last_epoch+1, last_epoch+1+epoch):
    last_epoch = e
    tr_loss = 0
    bicubic_tr_loss = 0
    for T in tr_tensor:
        with tf.GradientTape() as tape:
            _,loss_tr_batch, bicubic_loss=model.call(T,
                                                     downSample = True,
                                                     training= True)
        tr_loss += loss_tr_batch.numpy()/tr_num
        bicubic_tr_loss += bicubic_loss.numpy()/tr_num
        grad = tape.gradient(loss_tr_batch, model.trainable_variables)
        optimizer.apply_gradients(
            grads_and_vars = zip(grad, model.trainable_variables))
    val_loss = 0
    bicubic_val_loss = 0
    for V in val_tensor:
        _,val_loss, bicubic_loss_ = model.call(V, downSample=True)
        loss_val += val_loss.numpy()/val_num
        bicubic_loss_val += bicubic_loss_.numpy()/val_num
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    bicubic_tr_loss_list.append(bicubic_tr_loss)
    bicubic_val_loss_list.append(bicubic_val_loss)
