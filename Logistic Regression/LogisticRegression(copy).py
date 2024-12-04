import tensorflow as tf
import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = []
        self.accuracy_records = []
        self.loss_records = []
    

    def fit(self, train_x, train_y):
        ones = np.ones([train_x.shape[0], 1])
        train_x = tf.constant(np.concatenate((train_x, ones), axis=1), dtype=tf.float32)
        self.weights = tf.Variable(np.random.randn(train_x.shape[1], 1), dtype=tf.float32)
        train_y = tf.cast(tf.reshape(train_y, (-1, 1)), dtype=tf.float32)

        for _ in range(self.iterations):
            with tf.GradientTape() as tape:
                y_pred = tf.matmul(train_x, self.weights)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_y, logits=y_pred))
        
            dloss_dw = tape.gradient(loss, self.weights)
            self.weights.assign_sub(self.learning_rate * dloss_dw)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(tf.nn.sigmoid(y_pred) >= 0.5, 1., 0.), train_y), dtype=tf.float32))
            
            if _ % 10 == 0:
                print(f"Iter{_+1}           Accuracy: {accuracy}")
            self.accuracy_records.append(accuracy)
            self.loss_records.append(loss)


LR = LogisticRegression(3000, 0.001)
data = pd.read_csv("S:\\Gits files\\Standardized dataset.csv")
train_x = data.iloc[:, 0:2]
train_y = data.iloc[:, -1]
LR.fit(train_x.values, train_y.values)