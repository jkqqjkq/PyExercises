import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sess = tf.InteractiveSession()
train_path = '/Users/apple/Desktop/Kaggle/DigitRecognizer/train.csv'
test_path = '/Users/apple/Desktop/Kaggle/DigitRecognizer/test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
scaler = MinMaxScaler()
X_train = train.drop('label',1)
X_train = scaler.fit_transform(X_train)
Y_train = train['label'].astype(np.float32)
Y_train = Y_train.reshape((-1,1))
X_test = test
Y_train = OneHotEncoder().fit_transform(Y_train).todense()
X_train[0]
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
def get_batch(X,Y,step,batch_size):
    for i in range(step // batch_size):
        batch_x = X[i*batch_size:i*batch_size + batch_size]
        batch_y = Y[i*batch_size:i*batch_size + batch_size]
        yield batch_x,batch_y,i
tf.global_variables_initializer().run()
for bx,by,i in get_batch(X_train,Y_train,Y_train.shape[0],25):
    #print bx.shape ,by.shape
    train_step.run(feed_dict={x:bx,y:by,keep_prob:0.9})
    if i % 100 == 0:
        train_acc = accuracy.eval(feed_dict={x:bx,y:by,keep_prob:1.0})
        print 'step %d, trainging acc %g' %(i,train_acc)
X_test_ = scaler.transform(X_test)
prediction = tf.argmax(y_conv,1)
result = []
for i in range(len(X_test_//50)):
    result.append(sess.run(prediction,feed_dict={x: X_test_[i*50:i*50+50],keep_prob:1.0}))
result1 = np.zeros(28000)
for i in range(result1.shape[0]):
    result1[i] = result[i//50][i%50]
cnn_sub = pd.DataFrame({'ImageId':range(1,28001),'Label':result1})
cnn_sub.to_csv('/Users/apple/Desktop/cnn_sub.csv')