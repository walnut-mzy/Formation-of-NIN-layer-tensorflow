import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers
import random
import os
def image_deals(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    # print(train_file.shape)
    # print(image_string)
    image_decoded = tf.image.decode_png(image_string)  # 解码png
    image_string = randoc(image_decoded)

    #image_resized = tf.image.resize(train_file, [160, 60]) / 255.0   #把图片转换为224*224的大小
    image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image, dtype=tf.float32) / 255.0-0.5
    # print(image)
    # print(image.shape)
    #print(image_resized,label)
    #image=randoc(image)
    return image
def randoc(train_file):
    int1=random.randint(1,10)
    if int1==1:
        train_file = tf.image.random_flip_left_right(train_file)   #左右翻折
    elif int1==2:
        train_file=tf.image.random_flip_up_down(train_file)
    return train_file
def train_test_get(train_test_inf):
    for root,dir,files in os.walk(train_test_inf, topdown=False):
        #print(root)
        #print(files)
        list=[root+"/"+i for i in files]
        print(root)
        return list,root[26:]
class NIN(keras.layers.Layer):
    # 自定义网络层
    def __init__(self,keras_size,input_chanl,output_chanl,padding,strides):
        super(NIN, self).__init__()
        # # 创建权值张量并添加到类管理列表中，设置为需要优化
        # self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)
        filters=64
        self.conv2d=keras.layers.Conv2D(filters=input_chanl,kernel_size=keras_size,padding=padding,strides=strides)
        self.relu=keras.layers.ReLU()
        self.conv2d1=keras.layers.Conv2D(filters=output_chanl,kernel_size=1,padding=padding,strides=strides)
        self.relu1=keras.layers.ReLU()
        self.conv2d2=keras.layers.Conv2D(filters=output_chanl,kernel_size=1,padding=padding,strides=strides)
        self.relu2=keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        x=self.conv2d(inputs)
        x=self.relu(x)
        x=self.conv2d1(x)
        x=self.relu1(x)
        x=self.conv2d2(x)
        x=self.relu2(x)
        #print(x)
        return x
class Myflatten(keras.layers.Layer):
    def __init__(self):
        super(Myflatten, self).__init__()
        self.flatten1=keras.layers.Flatten()
        self.reshape1=keras.layers.Reshape([62,1])
    def call(self, inputs, **kwargs):
        x=self.flatten1(inputs)
        x=self.reshape1(x)
        x=tf.squeeze(x,axis=0)
        #print(x)
        return x
if __name__ == '__main__':
    #创建数据集
    list1=[
        "0","1","2","3","4","5","6","7","8","9",
        "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","y","z",
        "A1","B1","C1","D1","E1","F1","G1","H1","I1","J1","K1","L1","M1","N1","O1","P1","Q1","R1","S1","T1","U1","V1","W1","Y1","Z1",
    ]
    train=[]
    label=[]
    num=0
    for i in list1:
        train0,label0=train_test_get("C:/Users/mzy/Desktop/机器学习/"+str(i))
        train+=train0
        for k in range(1000):
            #对label0进行处理

            label.append(num)
        num += 1
    label=tf.one_hot(label,depth=62)
    print(label.shape)
    # #查看数据集形状
    # train=tf.constant(train)
    # label=tf.constant(label)
    # print(train.shape)
    # print(label.shape)
    print(train[0])
    train=[image_deals(i) for i in train]
    train=tf.expand_dims(train,axis=1)
    print(train.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((train,label))
    train_dataset.shuffle(1000)
    train_dataset.batch(batch_size=100)
    #train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(train_dataset)

    criteon = losses.categorical_crossentropy
    # 网友的模型
    model = tf.keras.Sequential(
        [
            NIN(keras_size=11,input_chanl=96,output_chanl=96,padding="same",strides=4),
            tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding="same"),
            NIN(keras_size=5, input_chanl=256, output_chanl=256, padding="same",strides=2),
            tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding="same"),
            NIN(keras_size=3, input_chanl=384, output_chanl=384, padding="same",strides=1),
            tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding="same"),
            tf.keras.layers.Dropout(0.5),
            NIN(keras_size=4,input_chanl=62,output_chanl=62,padding="same",strides=1),
            tf.keras.layers.AveragePooling2D(padding="same"),
            Myflatten(),
        ]
    )
    model.build((None,28,28,1))
    model.summary()
    # categorical_crossentropy
    # 和
    # sparse_categorical_crossentropy
    # 都是交叉熵损失函数，使用哪种函数要根据标签的结构来选择
    #
    # 如果样本标签是one - hot编码，则用
    # categorical_crossentropy函数
    # 　　one - hot
    # 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
    # 如果样本标签是数字编码 ，则用sparse_categorical_crossentropy函数
    # 　　数字编码：2, 0, 1

    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss=losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    model.fit(train_dataset, batch_size=100,epochs=100)