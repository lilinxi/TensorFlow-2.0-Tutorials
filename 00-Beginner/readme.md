# Keras 机器学习基础知识

> Tensorflow 是 Google 开源的基于数据流图的机器学习框架，Keras是基于 Tensorflow 和 Theano 的深度学习库，是为了支持快速实践而对 Tensorflow 或者 Theano 的再次封装。Keras 已经被添加到 Tensorflow 中，成为其默认的框架，为 Tensorflow 提供更高级的 API。 

- **Keras常用数据集**

|数据集名称|keras简称|描述|
|---|---|---|
|CIFAR-10 小图像分类数据集|cifar10|50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 10 个类别。|
|CIFAR-100 小图像分类数据集|cifar100|50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 100 个详细类别，20 个粗略类别。|
|IMDB 电影评论情感分类数据集|imdb|数据集来自 IMDB 的 25,000 条电影评论，以情绪（正面/负面）标记。评论已经过预处理，并编码为词索引（整数）的序列表示。为了方便起见，将词按数据集中出现的频率进行索引，例如整数 3 编码数据中第三个最频繁的词。这允许快速筛选操作，例如：「只考虑前 10,000 个最常用的词，但排除前 20 个最常见的词」。作为惯例，0 不代表特定的单词，而是被用于编码任何未知单词。|
|路透社新闻主题分类数据集|reuters|数据集来源于路透社的 11,228 条新闻文本，总共分为 46 个主题。与 IMDB 数据集一样，每条新闻都被编码为一个词索引的序列（相同的约定）。|
|MNIST 手写字符分类数据集|mnist|训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类数字标签。|
|Fashion-MNIST 时尚物品分类数据集|fashion_mnist|训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类时尚物品标签。|
|Booston 房价回归数据集|boston_housing|样本包含 1970 年代的在波士顿郊区不同位置的房屋信息，总共有 13 种房屋属性。 目标值是一个位置的房屋房价的中位数（单位：k$）。|

## 图像分类

```
使用 Fashion MNIST 数据集进行图像分类
```

### 数据预处理

1. 均值转化

> mnist在进行预处理的时候也会将灰度值（0到255）除以255从而归一话到0到1的范围内。首先如果输入层 x 很大，在back propagation时传递到输入层的梯度就会很大，如果梯度非常大，学习率就必须非常小，否则会跳过local minimum。因此，学习率（学习率初始值）的选择需要参考输入层的数值，不如直接将数据归一化，这样学习率就不必再根据数据范围作调整。

2. 零均值化/中心化

> 让所有训练图片中每个位置的像素均值为0，使得像素值范围变为\[-128,127\]，以0为中心。这样做的优点是为了在反向传播中加快网络中每一层权重参数的收敛。原因一：把各个特征的尺度控制在相同的范围内，这样可以便于找到最优解；原因二：当x全为正或者全为负时，每次返回的梯度都只会沿着一个方向发生变化，即梯度变化的方向一会向上太多，一会向下太多。这样就会使得权重收敛效率很低。但当x正负数量“差不多”时，就能对梯度变化方向进行“修正”，使其接近直线方向，加速了权重的收敛。

### 模型

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 多分类模型，且不是 one-hot 类别
              metrics=['accuracy'])
```

- Adam 是目前最好的优化函数
- verbose：日志显示
- epochs：批次大小是一个超参数，用于定义在更新内部模型参数之前要处理的样本数量。时期数是一个超参数，它定义学习算法将在整个训练数据集中工作的次数。一个时期意味着训练数据集中的每个样本都有机会更新内部模型参数。一个时期由一个或多个批次组成。
- model 常用方法：compile(), fit(), predict(), evaluate()

## 文本分类

```
使用 IMDB 数据集进行情感分析
```

### TensorFlow Hub 文本分类

#### 数据预处理

使用**预训练文本嵌入（text embedding）模型**将文本转化为拥有约 1M 词汇量且维度固定的模型。

#### 模型

```python
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 二分类模型
              metrics=['accuracy'])
```

### 预处理文本分类

#### 数据预处理

1. 编码
    - The encoder encodes the string by breaking it into subwords or characters if the word is not in its dictionary. So the more a string resembles the dataset, the shorter the encoded representation will be.
2. 归一化
    - 长度标准化
    
编码 e.g.

```python
Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]
The original string: "Hello TensorFlow."

4025 ----> Hell
222 ----> o 
6307 ----> Ten
2327 ----> sor
4043 ----> Fl
2120 ----> ow
7975 ----> .
```

#### 模型

```python
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 二分类模型
              metrics=['accuracy'])
```

- Embedding：The resulting dimensions are: (batch, sequence, embedding). 嵌入层的输出是一个二维向量，每个单词在输入文本（输入文档）序列中嵌入一个。
- GlobalAveragePooling1D：降维

## 回归

```
使用 Auto MPG 数据集进行回归分析
```

### 数据预处理

- 使用 pandas 进行数据预处理
    - dropna()，去除空值
    - pop('Origin')，提出 label
    - sample()，采样训练集
    - drop()，去除训练集，获得测试集
    - describe()，使用其中的均值和方差进行数据规范化
    - transpose()
- 使用 seaborn 提供 matplotlib 之上的绘图 API

### 模型

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse',  # 平方均值误差
              optimizer=optimizer,
              metrics=['mae', 'mse'])  # 监控指标：绝对值均值误差，平方均值误差
```

检测当一定数量的 epochs 之后验证集的损失没有明显改善，则提前终止训练。通过定义一个callback来实现。

```python
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
```

## 过拟合和欠拟合

```
使用 IMDB 数据集进行正则化和 dropout 的效果测试
```

### 数据预处理

将文本从不定长的索引模型转化为定长的 multi_hot 模型。

### 模型

通过model.fit()的返回值history获取训练过程中的训练集损失和验证集损失进行绘图分析。

使用不同参数个数的模型，应用L1，L2正则化的模型，应用 dropout 的模型进行比较。

## 保存和加载

```
使用 MNIST 数据集测试保存和加载
```

### 数据预处理

对图像进行降维（展平）和均值转化。

### 模型

可以使用训练好的模型而无需从头开始重新训练，或在打断的地方开始训练，以防止训练过程没有保存。

#### checkpoints

- 保存：tf.keras.callbacks.ModelCheckpoint/model.save_weights()
- 加载：model.load_weights()

#### model

- 保存：model.save()
- 加载：tf.keras.models.load_model()
- HDF5 只是保存了对象的配置来保存模型的架构，SavedModel 保存了整个执行图。

---

# 加载和预处理数据

## CSV

```
使用 Titanic 数据集加载和处理CSV数据
```

### 数据预处理

使用tf.data.experimental.make_csv_dataset()加载csv文件为dataset对象。

#### 连续值处理

将连续值拼接为一个矩阵，并进行零均值化。

#### 类别值处理

使用tf.feature_column.categorical_column_with_vocabulary_list()创建一个分类列。

#### 组合处理

```python
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
```

将对连续值和类别值的处理合并成一个预处理层，作为输入层的一部分。

### 模型

```python
model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
```

## Numpy

```
使用 MNIST 数据集加载和处理Numpy数据
```

### 数据预处理

1. np.load()加载npz文件
2. tf.data.Dataset.from_tensor_slices()组合数据和标签，形成Dataset对象
3. Dataset对象：shuffle(), batch()

## Pandas

```
使用 MNIST 数据集加载和处理Pandas数据
```

### 数据预处理

1. pd.read_csv() 加载csv数据
2. pd.Categorical(), df.thal.cat.codes 将 Pandas.DataFrame 类别信息转化成数值信息
    - Pandas.DataFrame 直接转化成Dataset对象
    - Pandas.DataFrame 先转化成 dict，再转化成Dataset对象，在model前加读取dict的输入层

### 模型

```python
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

## 图像

```
使用 flower_photos 数据集加载和处理图像数据
```

### 数据预处理

1. tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory()加载图像
2. 自主加载图像
    1. tf.data.Dataset.list_files()
    2. 从路径读取图像
    3. 转换图像
    4. 获取label
    5. prefetch and use cache(加速)


## 文本

```
使用《Homer's Illiad》的三种翻译版本进行训练，预测来自那种翻译版本
```

### 数据预处理

1. tf.data.TextLineDataset 建立文本数据集
2. tfds.features.text.Tokenizer() 或许词汇表数量
3. tfds.features.text.TokenTextEncoder() 编码句子
4. 划分为训练集和测试集，并将各个批次的文本向量补齐（引入了0来填充，词汇表数目加一）

### 模型

- RNN->BiRNN
- LSTM->BiLSTM
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64)) # 将整数表示转换为密集矢量嵌入
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))) # BiLSTM
for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))

model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

# references

- [tutorials](https://www.tensorflow.org/tutorials)
- [github tutorials(official)](https://github.com/tensorflow/docs)
- [github tutorials(official-fork)](https://github.com/lilinxi/docs)
- [github tutorials(official-zh)](https://github.com/tensorflow/docs/blob/master/site/zh-cn/tutorials)
- [github tutorials(official-en)](https://github.com/tensorflow/docs/blob/master/site/en/tutorials)
- [github tutorials(advanced code)](https://github.com/lilinxi/TensorFlow-2.x-Tutorials)