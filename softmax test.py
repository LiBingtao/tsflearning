import numpy as np
import tensorflow as tf

# Import data
train_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_data.csv', delimiter=",", skiprows=0)
train_target = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_target_oh.csv', delimiter=",", skiprows=0)

test_data= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_data.csv', delimiter=",", skiprows=0)
test_target= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_target_oh.csv', delimiter=",", skiprows=0)
  # Create the model
x = tf.placeholder(tf.float32, [None, 8])
W = tf.Variable(tf.zeros([8, 8]))
b = tf.Variable(tf.zeros([8]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 8])


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
  # Train
for _ in range(20000):
  sess.run(train_step, feed_dict={x: train_data, y_: train_target})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_data,y_: test_target}))


