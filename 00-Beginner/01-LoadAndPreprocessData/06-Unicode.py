import tensorflow as tf

print(tf.constant(u"Thanks 😊"))
print(tf.constant([u"You're", u"welcome!"]).shape)
# Unicode string, represented as a UTF-8 encoded string scalar.
text_utf8 = tf.constant(u"语言处理")
print(text_utf8)
# Unicode string, represented as a UTF-16-BE encoded string scalar.
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
print(text_utf16be)
# Unicode string, represented as a vector of Unicode code points.
# ord(): 它以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值
text_chars = tf.constant([ord(char) for char in u"语言处理"])
print(text_chars)

# TensorFlow 编码和解码

print(tf.strings.unicode_decode(text_utf8, input_encoding='UTF-8'))
print(tf.strings.unicode_encode(text_chars, output_encoding='UTF-8'))
print(tf.strings.unicode_transcode(text_utf8, input_encoding='UTF8', output_encoding='UTF-16-BE'))

# 批次维度

# A batch of Unicode strings, each represented as a UTF8-encoded string.
batch_utf8 = [s.encode('UTF-8') for s in [u'hÃllo', u'What is the weather tomorrow', u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding='UTF-8')
for sentence_chars in batch_chars_ragged.to_list():
    print(sentence_chars)

# 齐次化长度

batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded.numpy())

# SparseTensor：稠密存储稀疏矩阵
batch_chars_sparse = batch_chars_ragged.to_sparse()
print(batch_chars_sparse)
