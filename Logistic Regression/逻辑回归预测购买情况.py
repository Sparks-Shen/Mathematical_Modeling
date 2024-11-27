import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#导入数据
data =  pd.read_csv("C:\\Users\\沈德存\\Desktop\\Mathematical Modeling\\Logistic回归\\逻辑回归数据集.csv") # 400rows,5columns

data = data.sample(frac=1,random_state=10)#随机打乱

split_num = 100#划分数据集和训练集
train_set = data.iloc[:-split_num,:]
test_set = data.iloc[-split_num:,:]

train_x = train_set.iloc[:,1:-1]
test_x = test_set.iloc[:,1:-1]

train_gender = tf.constant(train_x["Gender"].apply(lambda x :1 if x=="Male" else 0).to_list(),dtype=tf.float32)

train_age_max = max(train_x["Age"])
train_age_min = min(train_x["Age"])
train_age_mean = np.mean(train_x["Age"])
train_age = tf.constant(train_x["Age"].to_list(),dtype=tf.float32)
#train_age = (train_age-train_age_min)/(train_age_max-train_age_min)#归一化
train_age = (train_age-train_age_mean)/np.std(train_age)

train_income_max = max(train_x["EstimatedSalary"])
train_income_min = min(train_x["EstimatedSalary"])
train_income_mean = np.mean(train_x["EstimatedSalary"])
train_income = tf.constant(train_x["EstimatedSalary"],dtype=tf.float32)
#train_income = (train_income-train_income_min)/(train_income_max-train_income_min)#归一化
train_income = (train_income-train_income_mean)/np.std(train_income)

ones = np.array([1]*(400-split_num))
input = tf.stack([train_gender,train_age,train_income,ones],axis=1)#300*4

w = tf.Variable([[np.random.randn()],
                 [np.random.randn()],
                 [np.random.randn()],
                 [np.random.randn()]],dtype=tf.float32)


#超参数定义
train_y = tf.constant(train_set["Purchased"].to_list(),shape=[300,1],dtype=tf.float32)
epoch = 8000
lr = 0.00005
total_loss_keep = []
accr_keep = []
for _ in range(epoch):
    with tf.GradientTape() as tape:
        z = tf.matmul(input,w)
        esti = tf.sigmoid(z)
        a = -train_y*tf.math.log(esti)
        b = - (1-train_y)*tf.math.log(1.-esti)
        loss = a + b
    dl_dw = tape.gradient(loss,w)
    w.assign_sub(lr*dl_dw)
    accr = tf.reduce_mean(tf.cast(tf.equal(tf.where(esti>=0.5,1.,0.),train_y),dtype=tf.float32))
    accr_keep.append(accr)
    total_loss_keep.append(tf.reduce_mean(loss))
    print("epoch{}     预测概率:{}".format(_,accr))

plt.plot(np.arange(epoch),accr_keep)
plt.title("Accuracy rate")
plt.show()

plt.plot(np.arange(epoch),total_loss_keep)
plt.title("Loss")
plt.show()
#测试
test_gender = tf.constant(test_x["Gender"].apply(lambda x :1 if x=="Male" else 0).to_list(),dtype=tf.float32)

test_age = tf.constant(test_x["Age"].to_list(),dtype=tf.float32)
#test_age = (test_age-train_age_min)/(train_age_max-train_age_min)#归一化
test_age = (test_age - tf.reduce_mean(test_age))/tf.math.reduce_std(test_age)

test_income = tf.constant(test_x["EstimatedSalary"],dtype=tf.float32)
#test_income = (test_income-train_income_min)/(train_income_max-train_income_min)#归一化
test_income = (test_income - tf.reduce_mean(test_income))/tf.math.reduce_std(test_income)


ones = np.array([1]*split_num)
test_input = tf.stack([test_gender,test_age,test_income,ones],axis=1)
z = tf.matmul(test_input,w)
esti = tf.sigmoid(z)

test_y = tf.constant(test_set["Purchased"].to_list(),shape=[split_num,1],dtype=tf.float32)

is_right = tf.equal(tf.where(esti>=0.5,1.,0.),test_y)
accr = tf.reduce_mean(tf.cast(is_right,dtype=tf.float32))
#for esti,real in zip(esti,test_y):
#    print("esti:",esti.numpy(),"    real:",real.numpy())
print("测试集的准确率：",accr)
print("W:",w.numpy())