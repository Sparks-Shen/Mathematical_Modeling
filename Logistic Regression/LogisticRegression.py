"""
Created on December 3 16:01:45 2024

@author: Sparks_Shen
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


# Welcome to the LogisticRegression model (Solving the problem of binary classification prediction)


class LogisticRegression:


    def instruction(self):
        print(
            "You can use 'import LogisticRegression as LR'. \n",
            "LogisticRefression: Your input should be the PATH of your file for LR_training. \n",
            "The file you input must be encoded by 'utf-8'. \n",
            "LogisticRegression has separated 'features' and 'tag' from the file you input automatically. You can view them by printing '.features' and '.tag'. \n",
            "Remember that your data['tag'] must be 0/1. \n"
        )




    def __init__(self, path, file_type):   # 'path' is the PATH of your input file
        if (file_type == "csv"):
            self.data = pd.read_csv(path, encoding='utf-8')
        elif (file_type == "xlsx"):
            self.data = pd.read_excel(path)

        else : print("The form of your file is not supported. ")
        self.features = []
        self.tag = None
        self.weights = []
        self.bias = []


        for i in range(0, len(self.data.columns)-1):    # determine the index of features and tag
            self.features.append(self.data.columns[i])
        self.tag = self.data.columns[-1]
        self.ones = np.ones(self.data.shape[0])
        self.data["ones"] = self.ones.reshape(-1, 1)
        self.features.append("ones")


        # bia has been added into matrix-'weight', so it doesn't need to be initialized here again
        # self.weights = tf.Variable(
        #     tf.random.normal([len(self.features), 1]), dtype=tf.float32
        # )
        # self.weights = tf.reshape(self.weights, (len(self.features), 1))
        # self.weights = tf.Variable(self.weights)

        self.weights = tf.Variable(np.random.randn(len(self.features), 1),dtype=tf.float32)




    def fit(self, iterative, learning_rate):
        self.iter = iterative
        self.lr = learning_rate
        self.loss_record = []
        self.accuracy_record = []


        x = tf.convert_to_tensor(self.data[self.features].values, dtype=tf.float32)
        y = tf.convert_to_tensor(self.data[self.tag].values, dtype=tf.float32)
        y = tf.reshape(y, (-1, 1))
        w = self.weights
        for i in range(self.iter):
            with tf.GradientTape() as tape:
                result = tf.matmul(x, w)
                pred = tf.nn.sigmoid(result)    # pred = 1 / 1+e^(-result)
                a = - y * tf.math.log(pred)
                b = - (1-y) * tf.math.log(1.-pred)
                loss = - y * tf.math.log(pred) - (1-y) * tf.math.log(1.-pred)   # Cross entropy loss function
            dloss_dw = tape.gradient(loss, w)
            w.assign_sub(self.lr * dloss_dw)     # update weights

            
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred >= 0.5, 1., 0.), self.data[self.tag]), dtype = tf.float32))
            print(accuracy.numpy())

            self.accuracy_record.append(accuracy)
            self.loss_record.append(tf.reduce_mean(loss))
            print(f"iter{i+1}         prediction_accuracy: {accuracy}")




    def visualize(self):
        plt.plot(np.arange(self.iter), self.accuracy_record)
        plt.title("Accuracy Record")
        plt.show()

        plt.plot(np.arange(self.iter), self.loss_record)
        plt.title("Loss Record")
        plt.show()
        

LR = LogisticRegression("S:\\Gits files\\标准化的数据集.csv", "csv")
LR.fit(4000, 0.00005)
LR.visualize()