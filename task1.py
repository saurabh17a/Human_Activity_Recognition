#!/usr/bin/env python
# coding: utf-8

# In[1]:
from model_script import activity_name,N_EPOCHS,LEARNING_RATE,path

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
#get_ipython().run_line_magic('matplotlib', 'inline')

#sns.set(style='whitegrid', palette='muted', font_scale=1.5)

#rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42


# In[2]:


#columns = ['x-axis', 'y-axis','z-axis','xg-axis', 'yg-axis','zg-axis','activity']
#df1 = pd.read_csv('raj_task.csv', header = None, names = columns)
#df2 = pd.read_csv('raj_task_1.csv', header = None, names = columns)
#df3 = pd.read_csv('raj_walking.csv', header = None, names = columns)
#df4 = pd.read_csv('raj_sitting.csv', header = None, names = columns)
#df5 = pd.read_csv('dataset/new_action.csv', header = None, names = columns)
#df = pd.concat([df1, df2, df3,df4], ignore_index= True)
#df  = df.iloc[1:]
#df = df.dropna()


# In[3]:


#df.head()
df = pd.read_csv(path)

# In[4]:


tf.__version__


# In[5]:


df.shape


# In[6]:


df['activity'].value_counts().plot(kind='bar', title='Training examples by activity type');


# In[7]:


N_TIME_STEPS = 500
N_FEATURES = 6
step = 100
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].values[i: i + N_TIME_STEPS]
    ys = df['y-axis'].values[i: i + N_TIME_STEPS]
    zs = df['z-axis'].values[i: i + N_TIME_STEPS]
    xsg = df['xg-axis'].values[i: i + N_TIME_STEPS]
    ysg = df['yg-axis'].values[i: i + N_TIME_STEPS]
    zsg = df['zg-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    #print label
    segments.append([xs, ys, zs, xsg, ysg, zsg])
    labels.append(label)
    #print labels


# In[8]:


reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)


# In[9]:


reshaped_segments.shape


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)


# In[11]:


X_train.shape


# In[12]:


X_test.shape


# In[13]:


N_CLASSES = 3
N_HIDDEN_UNITS = 64


# In[14]:


def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    
    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden =tf.split(hidden, N_TIME_STEPS, 0)

    # Stack 2 LSTM layers
    
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']


# In[15]:


#tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])


# In[16]:


pred_Y = create_LSTM_model(X)

pred_softmax = tf.nn.softmax(pred_Y, name="y_")


# In[17]:


L2_LOSS = 0.0015

l2 = L2_LOSS *     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_Y, labels = Y)) + l2


# In[18]:


#LEARNING_RATE = 0.0025

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


# In[19]:


#N_EPOCHS = 20
BATCH_SIZE = 128


# In[20]:


saver = tf.train.Saver()

history = dict(train_loss=[], 
                     train_acc=[], 
                     test_loss=[], 
    
                     test_acc=[])

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_count = len(X_train)

for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                       Y: y_train[start:end]})

    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
                                            X: X_train, Y: y_train})

    _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
                                            X: X_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    #if i != 1 and i % 10 != 0:
        #continue

    print("epoch " + str(i) + ":  "+ "test accuracy:" + " " + str(acc_test) + " " + "loss:" + str(loss_test))
    
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

#print()
print("final results: accuracy:   " + str(acc_final) + "\t" +  "loss:  " +str(loss_final))


# In[21]:


pickle.dump(predictions, open("predictions_task1.p", "wb"))
pickle.dump(history, open("history_task1.p", "wb"))
tf.train.write_graph(sess.graph_def, '.', './checkpoint/activity_task1.pbtxt')  
saver.save(sess, save_path = "./checkpoint/activity_tesk1.ckpt")
sess.close()


# In[22]:


history = pickle.load(open("history_task1.p", "rb"))
predictions = pickle.load(open("predictions_task1.p", "rb"))


# In[23]:


plt.figure(figsize=(12, 8))

plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

plt.plot(np.array(history['test_loss']), "r-", label="Test loss")
plt.plot(np.array(history['test_acc']), "g-", label="Test accuracy")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()


# In[24]:


#activity
LABELS = ['task1','walking','sitting']


# In[25]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


# In[26]:


#max_test = np.argmax(y_test, axis=1)
#ls = list(max_test)
#ls


# In[27]:


max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)

max_test1 = max_test.tolist()
print (max_test1)
print (type(max_test1))
max_predictions1 = max_predictions.tolist()
print(max_predictions1)

confusion_matrix(max_test1, max_predictions1)


# In[28]:


def data_reshaping(df):
    #df['activity'].value_counts().plot(kind='bar', title='Testing examples by activity type');
    N_TIME_STEPS = 500
    N_FEATURES = 6
    step = 100
    segments = []
    test_labels = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = df['x-axis'].values[i: i + N_TIME_STEPS]
        ys = df['y-axis'].values[i: i + N_TIME_STEPS]
        zs = df['z-axis'].values[i: i + N_TIME_STEPS]
        xsg = df['xg-axis'].values[i: i + N_TIME_STEPS]
        ysg = df['yg-axis'].values[i: i + N_TIME_STEPS]
        zsg = df['zg-axis'].values[i: i + N_TIME_STEPS]
        
        label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
        segments.append([xs, ys, zs, xsg, ysg,zsg])
        test_labels.append(label)
    #print test_labels
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
    #test_labels = np.asarray(pd.get_dummies(test_labels), dtype = np.float32)
    #X_train, X_test, y_train, y_test = train_test_split(
        #reshaped_segments, test_labels, test_size=0.99, random_state=RANDOM_SEED)
    X_test = reshaped_segments
    y_test = test_labels
    
    #print test_labels
    return X_test, y_test
    


# In[29]:

"""
test = pd.read_csv('test_raj.csv', header = None, names = columns)


# In[30]:


x_test1, y_test1 = data_reshaping(test)


# In[ ]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph("./checkpoint/activity_tesk1.ckpt.meta")
    saver.restore(sess,tf.train.latest_checkpoint("./checkpoint"))
    predictions1= sess.run(pred_softmax, feed_dict={X: x_test1})
    #predictions, acc_final, loss_final = sess.run([pred_softmax, a, feed_dict={X: X_test1, Y: y_test1})
    sess.close()


# In[ ]:


y_ls = list(y_test1)
print(y_ls)


# In[ ]:


max_predictions1 = np.argmax(predictions1, axis=1) + 1
print(list(max_predictions1))


# In[ ]:





# In[ ]:


columns = ['time','x-axis', 'y-axis','z-axis','xg-axis', 'yg-axis','zg-axis','activity']
df_time = pd.read_csv('test_raj_time.csv', header = None, names = columns)


# In[ ]:


time = df_time['time']


# In[ ]:


single_task_list = []


# In[ ]:


pred_list = list(max_predictions1)
activity = []
temp = ['activity name','start time','End time','Accuracy']
activity.append(temp)


# In[ ]:


i = 400
for items in pred_list:
    i+=100
    if len(single_task_list) == 6:
        if single_task_list.count(1) >= single_task_list.count(2):
            accuracy = single_task_list.count(1) *100 / 6
            ls = []
            ls.append(1)
            ls.append(time[i-500])
            ls.append(time[i])
            ls.append(accuracy)
            activity.append(ls)
        single_task_list =[]
    else:
        single_task_list.append(items)
        


# In[ ]:


for items in activity:
    print (items)


# In[ ]:





# In[ ]:




"""