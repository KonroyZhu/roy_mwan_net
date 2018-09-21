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

    def mat_weight_mul(self, mat, weight):
        """ 带batch的矩阵相乘"""
        # mat(b,n,m) *weight(m,p)
        # => (b,n,m) -> (b*n,m)
        # => (b*n,m)*(m,p)=(b*n,p)
        # return (b,n,p)
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])  # 检查矩阵是否可以相乘
        mat_reshape = tf.reshape(mat, shape=[-1, mat_shape[-1]])  # [b*n,m]
        mul = tf.matmul(mat_reshape, weight)  # [b*n,p]
        return tf.reshape(mul, shape=[-1, mat_shape[1], weight_shape[-1]])  # [b,n,p]

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal(shape=[dim]), name=name)

    def __init__(self):
        self.opts=json.load(open("model/config.json"))
        opts=self.opts
        self.embedding_matrix=self.random_weight(dim_in=opts["vocab_size"],dim_out=opts["embedding_size"])

        # multi way attention weigth
        # concatenate
        self.W_c_p = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_att_w_p")
        self.W_c_q = self.random_weight(dim_in=2 * opts["hidden_size"], dim_out=opts["hidden_size"],
                                        name="concat_att_w_q")
        self.V_c = self.random_bias(dim=opts["hidden_size"], name="concat_att_v")
        # bilinear
        self.W_b = self.random_weight(dim_in=2*opts["hidden_size"],dim_out=2*opts["hidden_size"],name="bilinear_att_w")
        # dot
        self.W_d=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="dot_att_w")
        self.V_d=self.random_bias(dim=opts["hidden_size"],name="dot_att_v")
        # minus
        self.W_m=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="minus_att_w")
        self.V_m=self.random_bias(dim=opts["hidden_size"],name="minus_att_v")
        # self att
        self.W_s=self.random_weight(dim_in=2*opts["hidden_size"],dim_out=opts["hidden_size"],name="self_att_w")
        self.V_s = self.random_bias(dim=opts["hidden_size"], name="minus_att_s")


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
            print("obtaining concat attention...")
            """adapted from pytorch
           _s1 = self.Wc1(hq).unsqueeze(1) # (b,q,2h) * (2h,h) = (b,p,h) =us1= (b,1,q,h)
           _s2 = self.Wc2(hp).unsqueeze(2) # (b,p,2h) * (2h,h) = (b,p,h) =us2= (b,p,1,h)
           sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze() # 自动广播(2,3维度) (b,p,q,h) * (h,1) = (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtc = ait.bmm(hq) # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s1=self.mat_weight_mul(h_q, self.W_c_q)
            _s1=tf.expand_dims(_s1,axis=1) # (b,1,q,h)
            _s2=self.mat_weight_mul(h_p, self.W_c_p)
            _s2=tf.expand_dims(_s2,axis=2) # (b,p,1,h)
            tanh=tf.tanh(_s1+_s2) # (b,p,q,h) 在维度为1的位置上自动广播 相当于til操作

            # sjt=tf.squeeze(tf.matmul(tanh,tf.reshape(self.V_c,shape=[-1,1]))) # (b,p,q,h) * (h,1) =sq=> (b,p,q) TODO: tf 中(b,p,q,h) * (h,1) 不可直接matmul
            sjt=tf.matmul(tf.reshape(tanh,[-1,opts["hidden_size"]]),tf.reshape(self.V_c,[-1,1]))# (b*p*q,h) * (h,1) => (b*p*q,1)
            sjt=tf.squeeze(tf.reshape(sjt,shape=[opts["batch"],opts["p_len"],opts["q_len"],-1])) # (b,p,q,1) =sq=> (b,p,q)
            ait=tf.nn.softmax(sjt,axis=2) # (b,p,q)

            # apply attention weight
            qtc= tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (b,p,2h) 当识别到有batch时 tf.matmul 自动转变为keras.K.batch_dot

            print("_s1: {} | _s2: {}".format(_s1, _s2))
            print("tanh:", tanh, "自动广播")
            print("sjt: {}".format(sjt))
            print("qtc: {}".format(qtc))
            # Bi-linear Attention
            print("obtaining bi-linear attention...")
            """adapted from pytorch
           _s1 = self.Wb(hq).transpose(2, 1) # (b,q,2h) * (2h,2h) =trans= (b,2h,q)
           sjt = hp.bmm(_s1) # (b,p,2h) b* (b,2h,q) = (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtb = ait.bmm(hq) # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s = self.mat_weight_mul(h_q, self.W_b) # (b,q,2h) * (2h,2h) => (b,q,2h)
            _s = tf.transpose(_s,perm=[0,2,1]) # (b,q,2h) => (b,2h,q)
            sjt= tf.matmul(h_p,_s) # (b,p,2h) batch* (b,2h,q) => (b,p,q)
            ait=tf.nn.softmax(sjt,axis=2) # (b,p,q)
            qtb=tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (b,p,2h) 顺序不能反

            print("qtb: {}".format(qtb))

            # Dot Attention
            print("obtaining dot attention...")
            """ adapted from pytorch
           _s1 = hq.unsqueeze(1) # (b,q,2h) =us1= (b,1,q,2h)
           _s2 = hp.unsqueeze(2) # (b,p,2h) =us2= (b,p,1,2h)
           sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze() # (b,p,q,2h)*(2h,h)*(h,) =sq= (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtd = ait.bmm(hq)  # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s1 = tf.expand_dims(h_q,axis=1) # (b,q,2h) => (b,1,q,2h)
            _s2 = tf.expand_dims(h_p,axis=2) # (b,p,2h) => (b,p,1,2h)

            _tanh=self.mat_weight_mul(_s1 * _s2, self.W_d) # 乘法自动广播 (b,p,q,2h)*(2h,h) =mat_weight_mul=> (b,p*q,h)
            tanh=tf.reshape(_tanh,[opts["batch"],opts["p_len"],opts["q_len"],-1]) # (b,p,q,h)
            _sjt=self.mat_weight_mul(_tanh,tf.reshape(self.V_d,[-1,1])) # (b,p*q,1) =sq=> (b,p*q,1)
            sjt=tf.squeeze(tf.reshape(_sjt,[opts["batch"],opts["p_len"],opts["q_len"],-1])) # (b.p,q)
            ait=tf.nn.softmax(sjt,axis=2)

            qtd = tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (b,p,2h)
            print("_tanh from mat_weight_mul: {}".format(_tanh))
            print("tanh reshaped: {}".format(tanh))
            print("_sjt from mat_weight_mul: {} ".format(_sjt))
            print("sjt reshaped: {}".format(sjt))
            print("qtd: {}".format(qtd))

            # Minus Attention
            print("obtaining minus attention...")
            """adapted from pytorch
           _s1 = hq.unsqueeze(1) # (b,1,q,2h)
           _s2 = hp.unsqueeze(2) # (b,p,1,2h)
           sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze() # (b,p,q,2h)*(2h,h)(h,) =sq= (b,p,q)
           ait = F.softmax(sjt, 2) # (b,p,q)
           qtm = ait.bmm(hq) # (b,p,q) b* (b,q,2h) = (b,p,2h)
           """
            _s1 = tf.expand_dims(h_q, axis=1)  # (b,q,2h) => (b,1,q,2h)
            _s2 = tf.expand_dims(h_p, axis=2)  # (b,p,2h) => (b,p,1,2h)

            _tanh=self.mat_weight_mul(_s1-_s2,self.W_m) # (b,p*q,h)
            _sjt=self.mat_weight_mul(_tanh,tf.reshape(self.V_m,[-1,1])) # (b,p*q,1)
            sjt=tf.squeeze(tf.reshape(_sjt,[opts["batch"],opts["p_len"],opts["q_len"],-1])) # (b,p,q)
            ait=tf.nn.softmax(sjt,axis=2)
            qtm=tf.matmul(ait,h_q) # (b,p,q) batch* (b,q,2h) => (n,p,2h)

            print("qtm: {}".format(qtm))

            # Self Matching Attention
            print("obtaining self attention...")
            """adapted from pytorch
           _s1 = hp.unsqueeze(1) # (b,1,p,2h)
           _s2 = hp.unsqueeze(2) # (b,p,1,2h)
           sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze() # (b,p,p,2h)*(2h,h)*(h,) =sq= (b,p,p)
           ait = F.softmax(sjt, 2) # (b,p,p)
           qts = ait.bmm(hp) # (b,p,p) b* (b,p,2h) = (b,p,2h)
           """
            _s1=tf.expand_dims(h_p,axis=1) # (b,1,p,2h)
            _s2=tf.expand_dims(h_p,axis=2) # (b,p,1,2h)
            tanh=self.mat_weight_mul(_s1*_s2,self.W_s) # (b,p*p,h)
            sjt=self.mat_weight_mul(tanh,tf.reshape(self.V_s,[-1,1])) # (b,p*p,1)
            sjt=tf.squeeze(tf.reshape(sjt,[opts["batch"],opts["p_len"],opts["p_len"],-1])) # (b,p,p)
            ait=tf.nn.softmax(sjt,axis=2)
            qts=tf.matmul(ait,h_p) # (b,p,p) batch* (b,p,2h) => (b,p,2h)
            print("qts: {}".format(qts))

        with tf.variable_scope("Aggregate_Layer"):
            print("Layer3: Aggregate Layer")
            aggregate=tf.concat([h_p, qts, qtc, qtd, qtb, qtm],axis=2) # (b,p,12h)
            print("aggregate: {}".format(aggregate))

            fw_cells = [self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["dropout"]) for _
                        in range(2)]
            bw_cells = [self.DropoutWrappedLSTMCell(hidden_size=opts["hidden_size"], in_keep_prob=opts["dropout"]) for _
                        in range(2)]
            aggregate_rnn=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, aggregate, dtype=tf.float32, scope="q_encoding")


