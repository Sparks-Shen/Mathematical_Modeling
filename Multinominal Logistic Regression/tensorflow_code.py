import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris_train = pd.read_csv(
    "C:\\Users\\沈德存\\Desktop\\Mathematical Modeling\\Multinomial Logistic回归\\Iris Dataset.csv")


def standardize(x):
    x_mean = np.mean(x)
    return (x - x_mean) / np.std(x)

x1 = iris_train["sepal_length"]     # z-score标准化特征数据
x1 = standardize(x1)
x2 = iris_train["sepal_width"]
x2 = standardize(x2)
x3 = iris_train["petal_length"]
x3 = standardize(x3)
x4 = iris_train["petal_width"]
x4 = standardize(x4)


iris_train["species"] = iris_train["species"].replace({
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
})
bias = np.ones((iris_train.shape[0], 1))  # 生成全1列为偏置特征列
x_train = iris_train[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values    # 划分特征训练集与标签训练集
y_train = iris_train["species"].values
x_train = np.concatenate([bias, x_train], axis=1)


y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train = tf.one_hot(y_train, depth=3)      # 转换为三列的one-hot编码


w = tf.Variable(       # 初始化随机数值的系数矩阵
    np.random.randn(5,3),
    dtype=tf.float64)


# 设置超参数
lr = 0.0001
iter = 1000
loss_record = []
accuracy_record = []

#梯度下降
for i in range(1, iter+1):
    with tf.GradientTape() as tape:
        z = tf.matmul(x_train, w)
        pred = tf.cast(tf.nn.softmax(z), dtype=tf.float32)      #softmax函数将回归函数z的值映射到0-1区间
        loss = -y_train*tf.math.log(pred) -(1-y_train)*tf.math.log(1-pred)      #交叉熵损失函数
    dl_dw = tape.gradient(loss, w)

    temp = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y_train, axis=1))
    accuracy = tf.reduce_mean(tf.cast(temp, dtype=tf.float32))
    
    
    loss_record.append(tf.reduce_mean(loss))
    accuracy_record.append(accuracy)

    w.assign_sub(lr*dl_dw)
    if i%50 == 0:
        print(f"iter: {i}, accuracy: {accuracy}")
        

#可视化
plt.plot(np.arange(1, iter+1), loss_record)    #损失函数图像
plt.title("loss_record")
plt.xlabel("iterative")
plt.ylabel("Loss")
plt.show()

plt.plot(np.arange(1, iter+1), accuracy_record)    #准确率函数图像
plt.title("accuracy_record")
plt.xlabel("iterative")
plt.ylabel("Accuracy")
plt.show()