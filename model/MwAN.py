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
        # concatenate
        self.W_c_p = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_attention_weight_p")
        self.W_c_q = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_attention_weight_q")
        self.V_c = self.random_bias(dim=opts["hidden_size"], name="concat_attention_v")
        # bilinear
        self.W_b = self.random_weight(dim_in=2*opts["hidden_size"],dim_out=2*opts["hidden_size"],name="bilinear_weight")



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
            print("encoding q...")
            h_q,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, q_emb_us, dtype=tf.float32, scope="q_encoding")
            print("encoding p...")
            h_p,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, p_emb_us, dtype=tf.float32, scope="p_encoding")
            print("encoding a...")
            a_enc,_,_=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, a_emb_rs_us, dtype=tf.float32, scope="a_encoding")

            h_q=tf.nn.dropout(tf.stack(h_q,axis=1),keep_prob=opts["dropout"])
            h_p=tf.nn.dropout(tf.stack(h_p,axis=1),keep_prob=opts["dropout"])
            a_enc=tf.nn.dropout(tf.stack(a_enc,axis=1),keep_prob=opts["dropout"])
        print("p/q_enc:",h_q)
        print("a_enc:",a_enc)

        with tf.variable_scope("Multiway_Matching_Layer"):
            print("Layer2: Multi-way Matching Layer")
            # Concat Attention
            print("obtaining concat attention")
            q_k_c=[]
            for t in range(opts["p_len"]):
                h_p_t=h_p[:,t,:]
                h_p_t_tiled=tf.concat([tf.reshape(h_p_t,shape=[opts["batch"],1,-1])]*opts["q_len"],axis=1)
                w_c_p=self.mat_weigth_mul(h_p_t_tiled, self.W_c_p) # (b,p,2h) * (2h,h) => (b,p,h)
                w_c_q=self.mat_weigth_mul(h_q, self.W_c_q) # (b,q,2h) * (2h,h) => (b,q,h)
                s_t_c=tf.squeeze(self.mat_weigth_mul(tf.tanh(w_c_p+w_c_q),tf.reshape(self.V_c,shape=[-1,1]))) # (b,q,h)* (h,1) =squeeze=> (b,q)
                a_c=tf.nn.softmax(s_t_c,axis=1)
                # print("attention c:",a_c)
                a_c=tf.concat([tf.reshape(a_c,shape=[opts["batch"],-1,1])]*2*opts["hidden_size"],axis=2)
                h_q_att = tf.reduce_sum(tf.multiply(h_q, a_c), axis=1)
                q_k_c.append(h_q_att)
            q_k_c=tf.stack(values=q_k_c,axis=1)
            print("concat qk:",q_k_c)

            # Bi-linear Attention
            print("obtaining bi-linear attention")
            q_k_b = []
            for t in range(opts["p_len"]):
                h_p_t = tf.squeeze(h_p[:, t, :])  # (b,1,2h) => (b,2h)
                h_p_t_W_b = tf.matmul(h_p_t,self.W_b)  # (b,2h)
                # print("h_p_t_W_b:",h_p_t_W_b)
                h_q_trans=tf.transpose(h_q,perm=[0,2,1]) # (b,q,2h) => (b,2h,q)
                # s_t_c=tf.keras.backend.batch_dot(h_p_t_W_b,h_q_trans)
                s_t_c=tf.matmul(tf.expand_dims(h_p_t_W_b,axis=1),h_q_trans) # the same with keras.K.batch_dot (b,1,q)
                s_t_c=tf.squeeze(tf.transpose(s_t_c,perm=[0,2,1])) # (b,q,1) =squeeze=> (b,q)

                a_c = tf.nn.softmax(s_t_c, axis=1)
                # print("attention c:",a_c)
                a_c = tf.concat([tf.reshape(a_c, shape=[opts["batch"], -1, 1])] * 2 * opts["hidden_size"], axis=2)
                h_q_att =tf.reduce_sum(tf.multiply(h_q, a_c),axis=1)
                q_k_b.append(h_q_att)
            q_k_b = tf.stack(values=q_k_b, axis=1)
            print("bi-linear qk:", q_k_b)


            # Bilinear Attention
            # Dot Attention
            # Minus Attention

