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

## 文本分类

## 回归

## 过拟合和欠拟合

## 保存和加载

---

# 加载和预处理数据

---

# references

- [tutorials](https://www.tensorflow.org/tutorials)
- [github tutorials(official)](https://github.com/tensorflow/docs)
- [github tutorials(official-fork)](https://github.com/lilinxi/docs)
- [github tutorials(official-zh)](https://github.com/tensorflow/docs/blob/master/site/zh-cn/tutorials)
- [github tutorials(official-en)](https://github.com/tensorflow/docs/blob/master/site/en/tutorials)
- [github tutorials(advanced code)](https://github.com/lilinxi/TensorFlow-2.x-Tutorials)