import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\沈德存\\Desktop\\Mathematical Modeling\\Multinomial Logistic回归\\Iris Dataset.csv")
data['species'] = data['species'].replace({'setosa':0,
                                           'versicolor':1,
                                           'virginica':2})
#print(data)
data = data.sample(frac=1,random_state=111)

for i in range(4):
    mean = np.mean(data.iloc[:,i])
    std = np.std(data.iloc[:,i])
    data.iloc[:,i]= (data.iloc[:,i]-mean)/std
def consistent(x):
    return tf.cast(x,dtype=tf.float32)
#划分测试集，训练集
train_split = 30
train_set = data.iloc[:-30,:]
test_set = data.iloc[-30:,:]

train_x = train_set.iloc[:,0:4]
train_y = train_set.iloc[:,-1]

test_x = test_set.iloc[:,0:4]
test_y = test_set.iloc[:,-1]

ones = np.ones(train_x.shape[0]).reshape(train_x.shape[0],1)
train_x = np.concatenate([train_x,ones],axis=1)
train_x = consistent(train_x)
train_y = tf.one_hot(indices = train_y,depth = 3)

w = tf.Variable(np.random.randn(5,3),dtype=tf.float32)

#def get_right(x):
    #for i in range(x.shape[0]):


arr_keep=[]
loss_keep=[]
epoch = 1000
lr=0.001
for _ in range(epoch):
    with tf.GradientTape() as tape:
        z = train_x @ w
        predict = tf.nn.softmax(z)
        fore = -train_y * tf.math.log(predict)
        rear =  - (1 - train_y) * tf.math.log(1-predict)
        loss = fore + rear
    dl_dw = tape.gradient(loss,w)
    a = tf.argmax(predict,axis=1)
    b = tf.argmax(train_y,axis=1)
    accr = tf.reduce_mean(tf.cast(tf.equal(a,b),dtype=tf.float32))
    arr_keep.append(accr)
    w.assign_sub(lr*dl_dw)
    loss_keep.append(tf.reduce_mean(loss).numpy())
    if _ % 10 == 0:
        print("epoch{}:    ACCR:{}".format(_,accr))
plt.plot(np.arange(epoch),arr_keep)
plt.title("ACCR")
plt.show()

plt.plot(np.arange(epoch),loss_keep)
plt.title("Loss")
plt.show()

#测试部分

test_y = tf.one_hot(indices = test_y,depth = 3)

ones = np.ones(test_x.shape[0]).reshape(test_x.shape[0],1)
test_x = np.concatenate([test_x,ones],axis=1)
test_x = consistent(test_x)
z = train_x @ w
predict = tf.nn.softmax(z)
accr = tf.reduce_mean(tf.cast(tf.equal(a,b),dtype=tf.float32))
print("测试集准确率：",accr)