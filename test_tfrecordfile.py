import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import dots as d
dat = d.dataset("6_dots.tfrecords")
i = 0
for el in dat:
    i+=1
    if i > 3:
        break
    print(el)


