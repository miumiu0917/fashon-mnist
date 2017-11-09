# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from image_util import resize
from PIL import Image

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', """チェックポイント保存先""")
tf.app.flags.DEFINE_integer('max_step', 50000, """訓練回数""")
tf.app.flags.DEFINE_string('data_dir', './data', """訓練データパス""")
tf.app.flags.DEFINE_string('log_dir', 'logs/', """学習ログ出力先""")
tf.app.flags.DEFINE_boolean('eval', False, """評価モードフラグ""")
tf.app.flags.DEFINE_string('photo_file', "", """評価対象画像""")

FLAGS = tf.app.flags.FLAGS

LABEL2NAME = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
              4: "Coat", 5: "Sandal", 6: "Shirts", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}


def training():
  data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  x = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])
  kb = tf.placeholder(tf.float32, [])

  convolution_op = convolution(x, kb)
  loss_op = loss(convolution_op, y)
  train_op = train(loss_op)
  accuracy_op = accuracy(convolution_op, y)
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    saver = tf.train.Saver()
    load_checkpoint(sess, saver)
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    summary_op = tf.summary.merge_all()
    
    for i in range(FLAGS.max_step):
      images, labels = data.train.next_batch(32)
      _, _loss, _accuracy = sess.run((train_op, loss_op, accuracy_op), feed_dict={x: images, y: labels, kb: 0.5})
      if i % 10 == 0:
        print('step: %03d loss: %0.06f accuracy: %0.06f' % (i, _loss, _accuracy))
      if i % 500 == 0 or i == FLAGS.max_step - 1:
        test_image, test_label = data.test.next_batch(3000)
        _accuracy, _ = sess.run([accuracy_op, summary_op], feed_dict={x: test_image, y: test_label, kb: 1.0})
        print('Test accuracy: %s' % _accuracy)
        saver.save(sess, FLAGS.checkpoint_dir, global_step=i)
    

def convolution(images, keep_prob):
  batch_size = tf.shape(images)[0]
  reshape = tf.reshape(images, [batch_size, 28, 28, 1])
  
  output = tf.layers.conv2d(reshape, filters=32, kernel_size=5, strides=(2,2), padding='SAME')
  output = tf.nn.relu(output)

  output = tf.layers.conv2d(reshape, filters=64, kernel_size=5, padding='SAME')
  output = tf.nn.relu(output)
  
  output = tf.layers.max_pooling2d(output, pool_size=(2, 2), strides=1)

  output = tf.layers.dropout(output, rate=keep_prob)

  output = tf.layers.conv2d(reshape, filters=32, kernel_size=5, strides=(2,2), padding='SAME')
  output = tf.nn.relu(output)

  output = tf.layers.conv2d(reshape, filters=64, kernel_size=5, padding='SAME')
  output = tf.nn.relu(output)
  
  output = tf.layers.max_pooling2d(output, pool_size=(2, 2), strides=1)

  output = tf.layers.dropout(output, rate=keep_prob)

  output = tf.contrib.layers.flatten(output)

  output = tf.layers.dense(output, 512)
  output = tf.nn.relu(output)

  output = tf.layers.dropout(output, rate=keep_prob)

  output = tf.layers.dense(output, 10)
  output = tf.nn.softmax(output)
  return output


def load_checkpoint(sess, saver):
  if os.path.exists(FLAGS.checkpoint_dir + 'checkpoint'):
    print('restore parameters...')
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
  else:
    print('initilize parameters...')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)



def loss(logits, labels):
  loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=[1]))
  tf.summary.scalar('loss', loss)
  return loss


def train(loss):
  return tf.train.AdadeltaOptimizer().minimize(loss)


def accuracy(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  return accuracy


def evaluate():
  img = Image.open(FLAGS.photo_file)
  gray_img = img.convert('L')
  img = [int(e) / 255.0 for e in resize(gray_img).tobytes()]
  x = tf.placeholder(tf.float32, [None, 784])
  kb = tf.placeholder(tf.float32, [])
  convolution_op = convolution(x, kb)
  with tf.Session() as sess:
    saver = tf.train.Saver()
    load_checkpoint(sess, saver)
    logits = sess.run([convolution_op], feed_dict={x: [img], kb: 1.0})
    print(LABEL2NAME[np.argmax(logits[0])])
  


def main():
  if FLAGS.eval:
    evaluate()
  else:
    training()

if __name__ == '__main__':
  main()