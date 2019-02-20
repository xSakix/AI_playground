import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

x = tf.add(a, b)


writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())

with tf.Session() as ses:
    result = ses.run(x)
    print(result)

writer.close()