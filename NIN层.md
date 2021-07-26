# NIN层

## **简介：**

我们提出了一种新型的深度网络结构，称为“Network In Network”（NIN），它可以增强模型在感受野（receptive field）内对局部区域（local patches）的辨别能力。传统的卷积层使用线性滤波器来扫描输入，后面接一个非线性激活函数。而我们则构建了一些结构稍复杂的微型神经网络来抽象receptive field内的数据。

我们用多层感知器实例化微型神经网络，这是一种有效的函数逼近器。
特征图可以通过微型神经网络在输入上滑动得到，类似于CNN；接下来特征图被传入下一层。深度NIN可以通过堆叠上述结构实现。通过微型网络增强局部模型，我们就可以在分类层中利用所有特征图的全局平均池化层（GAP），这样更容易解释且比传统的全连接层更不容易过拟合。

mlpconv层更好地模型化了局部块，GAP充当了防止全局过度拟合的结构正则化器。

使用这两个NIN组件，我们在CIFAR-10、CIFAR-100和Svhn数据集上演示了最新的性能。

通过特征映射的可视化，证明了NIN最后一个mlpconv层的特征映射是类别的置信度映射，这就激发了通过nin进行目标检测的可能性。


[这是一篇中英对照的有关NIN论文的文章](https://blog.csdn.net/adong1976/article/details/101861218)

 

## **作用：**

有效的防止过拟合，提高了网络的正确性

### **一些网络层错误率少的组合方式：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/fe64ef122bb04f1e9f322b00c266d25e.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


## **tensorflow实现自定义层**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers
class NIN(keras.layers.Layer):
    # 自定义网络层
    def __init__(self,keras_size,input_chanl,output_chanl,padding):
        super(NIN, self).__init__()
        # # 创建权值张量并添加到类管理列表中，设置为需要优化
        # self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)
        filters=64
        self.conv2d=keras.layers.Conv2D(filters=input_chanl,kernel_size=keras_size,padding=padding)
        self.relu=keras.layers.ReLU()
        self.conv2d1=keras.layers.Conv2D(filters=output_chanl,kernel_size=1,padding=padding)
        self.relu1=keras.layers.ReLU()
        self.conv2d2=keras.layers.Conv2D(filters=output_chanl,kernel_size=1,padding=padding)
        self.relu2=keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        x=self.conv2d(inputs)
        x=self.relu(x)
        x=self.conv2d1(x)
        x=self.relu1(x)
        x=self.conv2d2(x)
        x=self.relu2(x)
        return x
```

## categorical_crossentropy 和 sparse_categorical_crossentropy 



categorical_crossentropy 和 sparse_categorical_crossentropy 都是交叉熵损失函数，使用哪种函数要根据标签的结构来选择

如果样本标签是one-hot编码，则用 categorical_crossentropy函数
　　one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
如果样本标签是数字编码 ，则用sparse_categorical_crossentropy函数
　　数字编码：2, 0, 1

```python
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
    def __init__(self,keras_size,input_chanl,output_chanl,padding):
        super(NIN, self).__init__()
        # # 创建权值张量并添加到类管理列表中，设置为需要优化
        # self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)
        filters=64
        self.conv2d=keras.layers.Conv2D(filters=input_chanl,kernel_size=keras_size,padding=padding)
        self.relu=keras.layers.ReLU()
        self.conv2d1=keras.layers.Conv2D(filters=output_chanl,kernel_size=1,padding=padding)
        self.relu1=keras.layers.ReLU()
        self.conv2d2=keras.layers.Conv2D(filters=output_chanl,kernel_size=1,padding=padding)
        self.relu2=keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        x=self.conv2d(inputs)
        x=self.relu(x)
        x=self.conv2d1(x)
        x=self.relu1(x)
        x=self.conv2d2(x)
        x=self.relu2(x)
        return x
if __name__ == '__main__':
    #创建数据集
    list1=[
        "0",#"1","2","3","4","5","6","7","8","9",
        # "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","y","z",
        # "A1","B1","C1","D1","E1","F1","G1","H1","I1","J1","K1","L1","M1","N1","O1","P1","Q1","R1","S1","T1","U1","V1","W1","Y1","Z1",
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

    train_dataset = tf.data.Dataset.from_tensor_slices((train,label))
    train_dataset.shuffle(1000)
    train_dataset.batch(batch_size=100)
    train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(train_dataset)

    criteon = losses.categorical_crossentropy
    # 网友的模型
    model = tf.keras.Sequential(
        [

            NIN(keras_size=11,input_chanl=4096,output_chanl=4096,padding="same"),
            NIN(keras_size=11, input_chanl=4096, output_chanl=2048, padding="same"),
            NIN(keras_size=5, input_chanl=4096, output_chanl=2048, padding="same"),
            NIN(keras_size=5, input_chanl=2048, output_chanl=62, padding="same")
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
```

## 利用NIN实现手写英语字体的检验

**这个英语手写字体是由python中image_cature模块生成获得。**

**部分图片如下：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/bc7c42b52a89430687486491fdba0edd.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/f6ca94df941a4be1b25334a7f4ba7590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/1f6d6507b82548ff90243a2e60915de0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/0028975987c34194b7c94fd152b452ea.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/ff7cd745e3ec4876b267c0bbb57aa8ff.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/5fd0dc0d1c6c468d9d0a6176d7abd203.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)




```python
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
```

#这个训练效果是很不好的只是为了引入NIN模块和我自己制作的英文字母数据集，**所以这个只是为了提供一种NIN模块的方法**#