import pickle
import sklearn
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import pandas as pd
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten
import sys

# TODO: Fill this in based on where you saved the training data
training_file = "train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']


def convert_RGB_GRY(data, keepdims):
    input_rgb = data
    input_gry = np.sum(data/3, axis=3, keepdims=keepdims)
    return input_rgb, input_gry


X_train_rgb, X_train_gry = convert_RGB_GRY(X_train, keepdims=True)

X_train = X_train_gry

X_train_normalized = (X_train - 128)/128

X_train = X_train_normalized

X_train, y_train = shuffle(X_train, y_train)


def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)
    # print("1st Layer shape:", x.get_shape())

    # Activation.
    x = tf.nn.relu(x)

    # Max Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='VALID', name='conv1')
    x1 = x
    # print("1st Layer after max pooling shape:", x1.get_shape())

    # Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x, b2)
    # print("2nd Layer shape:", x.get_shape())

    # Activation.
    x = tf.nn.relu(x)

    # Max Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='VALID', name='conv2')
    x2 = x
    # print("2nd Layer after max pooling shape:", x2.get_shape())

    # Layer 3: Convolutional. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 16, 400), mean=mu, stddev=sigma))
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)
    # print("3rd Layer shape:", x.get_shape())

    # Activation.
    x = tf.nn.relu(x)
    x3 = x

    # Layer 4: Flatten. Input = 5x5x16. Output = 400.
    flat_layer2 = flatten(x2)
    # print("After Flattening 2nd Layer post max pooling shape:",
    #      flat_layer2.get_shape())

    # Flatten x. Input = 1x1x400. Output = 400.
    flat_layer3 = flatten(x3)
    # print("After Flattening 3rd Layer after activation shape:",
    #      flat_layer3.get_shape())

    # Concat layer2flat and x. Input = 400 + 400. Output = 800
    x = tf.concat([flat_layer3, flat_layer2], 1)
    # print("After Concatenating Flat Layers shape:", x.get_shape())

    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    # print("Dropout Layer shape:", x.get_shape())

    # TODO: Layer 4: Fully Connected. Input = 800. Output = 43.
    W4 = tf.Variable(tf.truncated_normal(
        shape=(800, 43), mean=mu, stddev=sigma))
    b4 = tf.Variable(tf.zeros(43), name="b4")
    logits = tf.add(tf.matmul(x, W4), b4)
    # print("Logits shape:", logits.get_shape())

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
x_final_test = tf.placeholder(tf.float32, (None, 32, 32, 1))
x_final_graph = tf.placeholder(tf.float32, (None, 32, 32, 1))
keep_prob = tf.placeholder(tf.float32)  # probability to keep units
one_hot_y = tf.one_hot(y, 43)

logits = LeNet(x)

# predict new images


def pipeline(file):
    global X_final_test
    my_images = []
    for i, img in enumerate(glob.glob(file)):
        X_final_test_name.append(img)
        image = cv2.imread(img)
        # print('Input Image ' + str(i) + ' ' + str(image.shape))
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        my_images.append(image)
    my_images = np.asarray(my_images)
    my_images_gry = np.sum(my_images/3, axis=3, keepdims=True)
    processed_image = (my_images_gry - 128)/128
    # print('After Processing Image: ' + str(processed_image.shape))

    return processed_image


# fetch only sign names
label_signs = pd.read_csv('./signnames.csv').values[:, 1]

# reading in an image
X_final_test = []
X_final_test_name = []
x_final_graph = []

fig, axs = plt.subplots(2, 3, figsize=(3, 2))
fig.subplots_adjust(hspace=.2, wspace=.2)
axs = axs.ravel()


softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


def main(folderName):
    X_final_test = pipeline('./' + folderName + '/*.png')
    X_final_graph = X_final_test

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./lenet.meta')
        saver.restore(sess, "./lenet")
        my_softmax_logits = sess.run(softmax_logits, feed_dict={
            x: X_final_test, keep_prob: 1.0})
        my_top_k = sess.run(top_k, feed_dict={x: X_final_test, keep_prob: 1.0})

        for i, image in enumerate(X_final_test):
            pred_label = my_top_k[1][i][0]
            index = np.argwhere(y_train == pred_label)[0]
            print('Predicted Label: {} \n {}'.format(
                pred_label, label_signs[pred_label]))


if __name__ == "__main__":
    folderName = sys.argv[1]
    main(folderName)
