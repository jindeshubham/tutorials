import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X = 2
X = np.vectorize(X)
Y = 3
Y = np.vectorize(Y)

result = np.dot(X,Y)

print result
# Y = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.05))
# print X


# v1 = tf.Variable(0.0)
# p1 = tf.placeholder(tf.float32)
# new_val = tf.add(v1,p1)
# update = tf.assign(v1,new_val)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(5):
#        print(sess.run(update,feed_dict={p1:1.0}))
#     print(sess.run(v1))
