import json
import math

import tensorflow as tf


class MwAN:
    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def DropoutWrappedLSTMCell(self, hidden_size, in_keep_prob, name=None):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True, name=name)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def mat_weigth_mul(self, mat, weight):
        """ 带batch的矩阵相乘"""
        # mat*weight => [batch_size,n,m] * [m,p]=[batch_size,p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])  # 检查矩阵是否可以相乘
        mat_reshape = tf.reshape(mat, shape=[-1, mat_shape[-1]])  # [batch_size*n,m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size*n,p]
        return tf.reshape(mul, shape=[-1, mat_shape[1], weight_shape[-1]])  # [batch_size,n,p]

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal(shape=[dim]), name=name)

    def __init__(self):
        self.opts=json.load(open("model/config.json"))
        opts=self.opts
        self.embedding_matrix=self.random_weight(dim_in=opts["vocab_size"],dim_out=opts["embedding_size"])

        # multi way attention weigth
        self.W_c_p = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_attention_weight_p")
        self.W_c_q = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_attention_weight_q")
        self.V_c = self.random_bias(dim=opts["hidden_size"], name="concat_attention_v")


    def build(self):
        print("building model...")
        opts=self.opts

        # placeholder
        #FIXME: batch should be changed to None
        query=tf.placeholder(dtype=tf.int32,shape=[opts["batch"],opts["q_len"]])
        para=tf.placeholder(dtype=tf.int32,shape=[opts["batch"],opts["p_len"]])
        ans=tf.placeholder(dtype=tf.int32,shape=[opts["batch"],3,opts["alt_len"]]) # 每个ans中有三个小句，第一句为正确答案 FIXME: alt_len should be None

        # embedding
        with tf.variable_scope("Embedding_Encoding_Layer"):
            print("Layer1: Embedding& Encoding Layer")
            q_emb=tf.nn.embedding_lookup(self.embedding_matrix,query) # (b,q,emb)
            p_emb=tf.nn.embedding_lookup(self.embedding_matrix,para) # (b,p,emb)
            a_emb=tf.nn.embedding_lookup(self.embedding_matrix,ans) # (b,3,a,emb)
            a_emb_rs=tf.reshape(a_emb,shape=[opts["batch"]*3,-1,opts["embedding_size"]]) # (3b,a,emb)
            print("p/q_emb:",q_emb)
            print("a_emb_reshaped:",a_emb_rs)

            q_emb_us=tf.unstack(q_emb,axis=1) # (q,b,emb)
            p_emb_us=tf.unstack(p_emb,axis=1) # (p,b,emb)
            a_emb_rs_us=tf.unstack(a_emb_rs,axis=1) # (a,3b,emb)

            fw_cells=[self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],in_keep_prob=opts["dropout"]) for _ in range(2)]
            bw_cells=[self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"],in_keep_prob=opts["dropout"]) for _ in range(2)]
            print("encoding...")
            q_enc,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, q_emb_us, dtype=tf.float32, scope="q_encoding")
            p_enc,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, p_emb_us, dtype=tf.float32, scope="p_encoding")
            a_enc,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, a_emb_rs_us, dtype=tf.float32, scope="a_encoding")

            q_enc=tf.nn.dropout(tf.stack(q_enc,axis=1),keep_prob=opts["dropout"])
            p_enc=tf.nn.dropout(tf.stack(p_enc,axis=1),keep_prob=opts["dropout"])
            a_enc=tf.nn.dropout(tf.stack(a_enc,axis=1),keep_prob=opts["dropout"])
        print("p/q_enc:",q_enc)
        print("a_enc:",a_enc)

        with tf.variable_scope("Multiway_Matching_Layer"):
            print("Layer2: Multi-way Matching Layer")
            # Concat Attention
            w_c_p=self.mat_weigth_mul(p_enc, self.W_c_p) # (b,p,2h) * (2h,h) => (b,p,h)
            w_c_q=self.mat_weigth_mul(q_enc, self.W_c_q) # (b,q,2h) * (2h,h) => (b,q,h)
            s_t_c=tf.squeeze(self.mat_weigth_mul(tf.tanh(w_c_p+w_c_q),tf.reshape(self.V_c,shape=[-1,1]))) # (b,q,h)* (h,1) =squeeze=> (b,q)
            a_c=tf.nn.softmax(s_t_c,axis=1)
            print("attention c:",a_c)

            # Bilinear Attention
            # Dot Attention
            # Minus Attention

