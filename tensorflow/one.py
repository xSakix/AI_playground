import tensorflow as tf

a = tf.add(2, 3)
print(a)
with tf.Session() as ses:
    print(ses.run(a))

print('-' * 80)

x = 2
y = 3

add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as ses:
    z, not_useless = ses.run([pow_op, useless])
    print(z)
    print(not_useless)