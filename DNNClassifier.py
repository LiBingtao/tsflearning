import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Import data
train_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/train_data.csv', delimiter=",", skiprows=0)
scaler.fit(train_data)
train_data = scaler.transform(train_data)
train_target = np.loadtxt('C:/Users/Bingtao LI/desktop/test.csv', delimiter=",", skiprows=0)
train_target = train_target.astype(int)[:,np.newaxis]

test_data = np.loadtxt('C:/Users/Bingtao LI/PycharmProjects/text/test_data.csv', delimiter=",", skiprows=0)
test_data = scaler.transform(test_data)
test_target = np.loadtxt('C:/Users/Bingtao LI/desktop/test2.csv', delimiter=",", skiprows=0)
test_target = test_target.astype(int)[:,np.newaxis]

def get_train_inputs():
  x = tf.constant(train_data)
  y = tf.constant(train_target)
  return x, y
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=8)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,10,10],n_classes=8)

classifier.fit(input_fn=get_train_inputs, steps=2000)


def get_test_inputs():
    x = tf.constant(test_data)
    y = tf.constant(test_target)
    return x, y


accuracy_score = classifier.evaluate(input_fn=get_test_inputs,steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))