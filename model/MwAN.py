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

    def __init__(self):
        self.opts=json.load(open("model/config.json"))
        opts=self.opts
        self.embedding_matrix=self.random_weight(dim_in=opts["vocab_size"],dim_out=opts["embedding_size"])


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

