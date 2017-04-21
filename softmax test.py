import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Import data
train_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_data.csv', delimiter=",", skiprows=0)
scaler.fit(train_data)
train_data = scaler.transform(train_data)
train_target = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_target_oh.csv', delimiter=",", skiprows=0)

test_data= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_data.csv', delimiter=",", skiprows=0)
test_data = scaler.transform(test_data)
test_target= np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_target_oh.csv', delimiter=",", skiprows=0)
  # Create the model

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
x = tf.placeholder(tf.float32, [None, 8])

W_fc1 = weight_variable([8, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

#W_fc3 = weight_variable([256, 30])
#b_fc3 = bias_variable([30])
#h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

#W_fc4 = weight_variable([30, 30])
#b_fc4 = bias_variable([30])
#h_fc4 = tf.matmul(h_fc3, W_fc4) + b_fc4

keep_prob = tf.placeholder(tf.float32)
h_fc4_drop = tf.nn.dropout(h_fc2, keep_prob)

W = weight_variable([512,8])
b = bias_variable([8])
y = tf.matmul(h_fc4_drop,W) + b

  # Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 8])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
  # Train
for i in range(5000):
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x: train_data, y_: train_target, keep_prob:1})
    print("step %d, training accuracy %g" % (i, train_accuracy))
  sess.run(train_step, feed_dict={x: train_data, y_: train_target, keep_prob:0.5})

# Test trained model

print(sess.run(accuracy, feed_dict={x: test_data,y_: test_target,keep_prob: 1.0}))


