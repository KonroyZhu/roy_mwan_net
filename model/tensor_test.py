import tensorflow as tf

p = 300
q = 30
h = 60
b = 20
hq = tf.get_variable(dtype=tf.int32, shape=[b, q, 2 * h], name="ques_enc")
hp = tf.get_variable(dtype=tf.int32, shape=[b, p, 2 * h], name="para_enc")

hq_ex = tf.expand_dims(hq, axis=1)
hp_ex = tf.expand_dims(hp, axis=2)
print("hq_ex:", hq_ex)
print("hp_ex:", hp_ex)

add = hq_ex + hp_ex  # (b,1,q,2h) + (b,p,1,2h) 自动广播 => （b,p,q,2h)
print("add:", add)  # (b,p,q,2h)

mul = hq_ex * hp_ex  # （b,1,q,2h)*(b,p,1,2h)
print("mul:", mul)  # （b,p,q,2h)

# matmul=tf.matmul(hq_ex,hp_ex) FIXME: shapes must be equal
multiply=tf.multiply(hp_ex,hp_ex)
print("multiply:", multiply)  # （b,p,1,2h)

