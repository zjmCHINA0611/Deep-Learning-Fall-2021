import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.metrics import *

class C2AE():
    def __init__(self,):
        # self.batch_size = 100
        self.batch_size = 70
        self.epoch = 100000
        self.alpha = 0.5
        self.lr = 1e-5
        # -5
        self.inputDimx = 6985
        # embedding size

        # self.feature_size = 512
        self.feature_size =512
        self.vec_style = "Word2vec"

        self.word2id = {}
        self.attention = False
        self.get_embedding()
        self.keep_prob = 0.7
        # 0.9
        self.std = 0.08
#糖尿病有关的疾病及其病发症
    def gene_diabetes(self,data):
        diaarr = np.zeros([1,data.shape[1]])
        list = [25000,
                25001,
                25002,
                25003,
                25010,
                25011,
                25012,
                25013,
                25020,
                25021,
                25022,
                25023,
                25030,
                25031,
                25032,
                25033,
                25040,
                24900,
                24901,
                24910,
                24911,
                24920,
                24921,
                24930,
                24931,
                24940,
                24941,
                24950,
                24951,
                24960,
                24961,
                24970,
                24971,
                24980,
                24981,
                24990,
                24991,
                25041,
                25042,
                25043,
                25050,
                25051,
                25052,
                25053,
                25060,
                25061,
                25062,
                25063,
                25070,
                25071,
                25072,
                25073,
                25080,
                25081,
                25082,
                25083,
                25090,
                25091,
                25092,
                25093,
                2535,
                3572,
                5881,
                64800,
                64801,
                64802,
                64803,
                64804,
                7751]
        print(list)
        list_ = [str(i) for i in list]
        with open('ICDCODEDICR.plk','rb') as f:
            ICD_dict = pickle.load(f)
        for ICD in list_:
            if ICD not in ICD_dict.keys():
                continue
            index = ICD_dict[ICD]
            for i in range(data.shape[0]):
                if data[i][index]!=0:
                    diaarr = np.concatenate([diaarr,[data[i]]],axis = 0)
        print(diaarr[1:,:].shape)
        return diaarr[1:,:]

        


    def get_data_max_length(self, true_data):

        max_length = 0
        data = true_data[:, :self.inputDimx]
        for x in data:
            li = [i for i, li in enumerate(x) if li == 1]
            if len(li) > max_length:
                max_length = len(li)
        print("max_length:", max_length)
        self.max_med_length = max_length

    def get_embedding(self,):
        vec = []
        # dic of ICD code
        word2id = {}
        root0 = "/Users/JinmingZhang/Downloads/EHR_papers_lyf/medical_data/MED_Att_multi/MIMIC_Word2Vec.vector"
        if self.vec_style == "Word2vec":
            f = open(root0, "r")

        content = f.readline()
        content = content.strip().split()
        dim = int(content[1])
        while True:
            content = f.readline()
            if content == "":
                break
            content = content.strip().split()
            word2id[content[0]] = len(word2id)
            content = content[1:]
            content = [(float)(i) for i in content]
            vec.append(content)
        f.close()
        word2id['UNK'] = len(word2id)
        word2id["BLANK"] = len(word2id)
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        vec = np.array(vec, dtype=np.float32)

        self.word_embedding = vec
        self.word2id = word2id

    def load_data_from_true_gene_word2vec(self,):
        # 最后的词向量嵌入用tensorflow实现
        # 真实数据
        root1 = "/Users/JinmingZhang/Downloads/EHR_papers_lyf/medical_data/MED_Att_multi/DIAGNOSE.npy"
        true_data = np.load(root1)
        # true_data = self.gene_diabetes(true_data)
        self.train_data_count = int(true_data.shape[0]*0.7)
        print('index size: ',true_data.shape)
        # 载入item_code_index
        root2 = "/Users/JinmingZhang/Downloads/EHR_papers_lyf/medical_data/MED_Att_multi/MIMIC_CODE_INDEX.plk"
        item_code_index = pickle.load(open(root2, "rb"))
        item_code_index = dict([(value, key) for key, value in item_code_index.items()])
        print("item_code_index:", item_code_index)

        self.get_data_max_length(true_data)
        ##############################################################################################################################################################
        # train_data_count = 10000
        x_train, y_train = true_data[:self.train_data_count, :self.inputDimx], true_data[:self.train_data_count,
                                                                          self.inputDimx:]
        x_test, y_test = true_data[self.train_data_count:, :self.inputDimx], true_data[self.train_data_count:,
                                                                             self.inputDimx:]

        ##############################################################################################################################################################

        def get_code_data(data_x=None, data_y=None):
            x_data = []
            y_data = []
            for j, x in enumerate(data_x):
                word2vec_index = []
                for i, code in enumerate(x):
                    if code == 1:
                        index = i
                        code_name = item_code_index[index]
                        if code_name not in self.word2id.keys():
                            continue
                        word2vec_index.append(self.word2id[code_name])
                # print("word2vec_index:",word2vec_index)

                if len(word2vec_index) > 0 and len(word2vec_index) <= self.max_med_length:
                    for i in range(self.max_med_length - len(word2vec_index)):
                        word2vec_index.append(self.word2id["BLANK"])
                    x_data.append(word2vec_index)
                    y_data.append(data_y[j])


            return x_data, y_data

        x_train, y_train = get_code_data(x_train, y_train)
        x_test, y_test = get_code_data(x_test, y_test)



        x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

        print("\nx_train.shape, x_test.shape:", x_train.shape, x_test.shape)
        print("\ny_train.shape, y_test.shape:", y_train.shape, y_test.shape)

        return x_train, x_test, y_train, y_test

    def init_Variables(self,shape):
        return (tf.truncated_normal(shape = shape,mean = 0,stddev = 0.1,dtype=tf.float64))

    def build_DCCA(self,X,Y):
        self.hidden_size = [512, 256]
        # for letent_size in (20,100,10)
        self.latent_size = 60

        with tf.variable_scope('DCCA'):
            W_1D = tf.get_variable(name = 'W_1D', shape = [self.feature_size,self.latent_size],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float64))
            # W_2D = tf.Variable(self.init_Variables([self.hidden_size[0],self.hidden_size[1]]))
            # W_3D = tf.Variable(self.init_Variables([self.hidden_size[1],self.latent_size]))

            b_1D = tf.get_variable(name = 'b_1D',shape = [self.latent_size],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float64))
            # b_2D = tf.Variable(self.init_Variables([self.batch_size, self.hidden_size[1]]))
            # b_3D = tf.Variable(self.init_Variables([self.batch_size, self.latent_size]))

            # X_DCCA_latent = tf.nn.leaky_relu(tf.matmul((tf.matmul((tf.matmul(X,W_1D) + b_1D),W_2D)+b_2D),W_3D)+b_3D)
            X_DCCA_latent = tf.nn.leaky_relu(tf.matmul(X,W_1D)+b_1D)
            X_DCCA_latent = tf.nn.dropout(X_DCCA_latent,keep_prob=self.keep_prob)
            # X_DCCA_latent = tf.layers.batch_normalization(X_DCCA_latent)


            self.hidden_size_1 = [800,500,200]
            W_1D_Y = tf.get_variable(name = 'W_1D_Y',shape = [self.label_size, self.hidden_size_1[1]],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            W_2D_Y = tf.get_variable(name = 'W_2D_Y',shape = [self.hidden_size_1[1], self.latent_size],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            # W_3D_Y = tf.Variable(self.init_Variables([self.hidden_size_1[1], self.hidden_size_1[2]]))
            # W_4D_Y = tf.Variable(self.init_Variables([self.hidden_size_1[2], self.latent_size]))


            b_1D_Y = tf.get_variable(name = 'b_1D_Y',shape = [self.hidden_size_1[1]],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            b_2D_Y = tf.get_variable(name = 'b_2D_Y',shape = [self.latent_size],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            # b_3D_Y = tf.Variable(self.init_Variables([self.batch_size, self.hidden_size_1[2]]))
            # b_4D_Y = tf.Variable(self.init_Variables([self.batch_size, self.latent_size]))

            # Y_DCCA_latent = tf.nn.leaky_relu(tf.matmul((tf.matmul((tf.matmul((tf.matmul(Y, W_1D_Y) + b_1D_Y), W_2D_Y) + b_2D_Y),W_3D_Y)+b_3D_Y),W_4D_Y)+b_4D_Y)
            Y1 = tf.matmul(Y, W_1D_Y) + b_1D_Y
            Y1 = tf.nn.dropout(Y1,keep_prob=self.keep_prob)
            Y2 = tf.matmul(Y1,W_2D_Y)+b_2D_Y
            Y2 = tf.nn.dropout(Y2, keep_prob=self.keep_prob)
            Y_DCCA_latent = tf.nn.leaky_relu(Y2)

            # Y_DCCA_latent = tf.layers.batch_normalization(Y_DCCA_latent)
            return X_DCCA_latent,Y_DCCA_latent,W_1D,b_1D,W_1D_Y,W_2D_Y

    def compute_correlation(self,X,Y):
        ''''r1,r2 regularization term'''
        r1 = 1e-4
        r2 = 1e-4
        H1 = tf.transpose(X)
        H2 = tf.transpose(Y)

        m = self.batch_size
        m = tf.cast(m,dtype = tf.float32)
        H1bar = H1 - 1.0/(m-1) * tf.matmul(H1,tf.ones([m,m],dtype = tf.float32))
        H2bar = H2 - 1.0/(m-1) * tf.matmul(H2,tf.ones([m,m],dtype = tf.float32))

        Sigmahat12 = 1.0/(m-1) * tf.matmul(H1bar, tf.transpose(H2bar))
        Sigmahat11 = 1.0/(m-1) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(self.latent_size)
        Sigmahat22 = 1.0/(m-1) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(self.latent_size)


        '''特征值分解算-1/2次幂'''

        [D1,V1] = tf.self_adjoint_eig(Sigmahat11)
        [D2,V2] = tf.self_adjoint_eig(Sigmahat22)

        SigmaHat11RootInv = tf.matmul(tf.matmul(V1,tf.diag(D1)**0.5),tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2,tf.diag(D2)**0.5),tf.transpose(V2))

        T = tf.matmul(tf.matmul(SigmaHat11RootInv,Sigmahat12),SigmaHat22RootInv)

        corr = tf.trace(tf.matmul(tf.transpose(T),T)) ** 0.5
        return -corr,Sigmahat11,Sigmahat22

    def auto_encoder(self,Y_latent,Y_initial):
        with tf.variable_scope('Autoencoder'):
            W_1A_Y = tf.get_variable(name ='W_1A_Y',shape = [self.latent_size, self.hidden_size[1]],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            W_2A_Y = tf.get_variable(name ='W_2A_Y',shape = [self.hidden_size[1],self.label_size],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            # 此处的第三个全一矩阵用来给Y加权做到多开不如不开
            W_3A_Y = tf.get_variable(name = 'weight',shape = [self.label_size,self.label_size],initializer=tf.ones_initializer(dtype=tf.float32))
            b_1A_Y = tf.get_variable(name ='b_1A_Y',shape = [self.hidden_size[1]],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            b_2A_Y = tf.get_variable(name ='b_2A_Y',shape = [self.label_size],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=self.std,dtype=tf.float32))
            Y1_auto = tf.nn.dropout((tf.matmul(Y_latent,W_1A_Y)+b_1A_Y),keep_prob=self.keep_prob)
            Y2_auto = tf.nn.dropout((tf.matmul(Y1_auto,W_2A_Y)+b_2A_Y),keep_prob=self.keep_prob)
            Y3_auto = tf.nn.dropout(tf.matmul(Y2_auto,W_3A_Y) , keep_prob=self.keep_prob)
            Y_decode = tf.layers.batch_normalization(Y3_auto)
            auto_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y_initial,logits = Y_decode))

            return Y_decode,auto_loss

    def fit(self,):
        x_train, x_test, y_train, y_test = self.load_data_from_true_gene_word2vec()

        self.label_size = y_train.shape[1]
        # X是索引 用tf实现词向量嵌入
        # X = tf.placeholder(tf.int32, shape=[None, None])
        X = tf.placeholder(tf.int32, shape=[None, None])
        Y = tf.placeholder(tf.float32, shape=[None,self.label_size])

        ###########################################################################################################
        ###########################################################################################################

        word_embedding = tf.get_variable(initializer=self.word_embedding, trainable=False, name="word_embedding")
        embedding_x = tf.nn.embedding_lookup(word_embedding, X)
        ###########################################################################################################
        ###########################################################################################################
        # embedding_size = tf.convert_to_tensor(tf.shape(embedding_x))
        # embedding_size = tf.cast(embedding_size,tf.float32)

        # 最后embedding的结果是output
        if self.attention is False:
            # output = embedding_x/tf.expand_dims(embedding_size,-1)
            output = tf.reduce_sum(embedding_x,1)

        # output_pre = tf.reduce_sum(embedding_x_pre, 1)
        X_DCCA_latent, Y_DCCA_latent,a,b,c,d = self.build_DCCA(output,Y)
        correlation,test1,test2 = self.compute_correlation(X_DCCA_latent,Y_DCCA_latent)
        y_pre,auto_loss = self.auto_encoder(Y_DCCA_latent,Y)

        # 预测
        # latent_pre,_ = self.build_DCCA(output_pre,Y_pre)
        # y_pre,_ = self.auto_encoder(latent_pre,Y_pre)

        # 同时优化两个模型的参数
        Weight_loss = tf.reduce_sum(tf.nn.relu(y_pre-Y))
        C2AE_loss = (1-self.alpha)*auto_loss + self.alpha * correlation*(1e-8) + Weight_loss*(1e-6)
        # C2AE_loss = (1 - self.alpha) * auto_loss + self.alpha * correlation * (1e-8)
        # -8,-8

        C2AE_loss_opt = tf.train.AdamOptimizer(self.lr).minimize(C2AE_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 每次选取batch大小的数据进行训练
        # print(self.word_embedding)
        nbatch = int(x_train.shape[0]/self.batch_size)
        for i in range(self.epoch):
            for j in range(nbatch):
                if j == nbatch-1:
                    j = 0
                batch_x = x_train[j*self.batch_size:(j+1)*self.batch_size,:]
                batch_y = y_train[j*self.batch_size:(j+1)*self.batch_size,:]
                #################################################################################################
                # print('loss:  ',np.array(sess.run(auto_loss, feed_dict={X: batch_x, Y: batch_y})).sum(axis = 0))
                # print(np.array(batch_y).sum(axis = 0))
                # print('X:',sess.run([test1,a,b],feed_dict={X: batch_x, Y: batch_y}),'\n\n')
                # print('X:',sess.run([Weight_loss],feed_dict={X: batch_x, Y: batch_y}),'\n\n')

                # print('Y:',sess.run([test2,c,d], feed_dict={X: batch_x, Y: batch_y}),batch_y,'\n')
                # print(sess.run(output,feed_dict={X: batch_x, Y: batch_y}))


                sess.run(C2AE_loss_opt, feed_dict={X : batch_x,Y: batch_y})


            if i%10 == 0:
                print('\nnum of epoch is: ', i)
                print('aoto:   ', np.array(sess.run(auto_loss, feed_dict={X: batch_x, Y: batch_y})).sum(axis=0))
                print('corr:   ', np.array(sess.run(correlation, feed_dict={X: batch_x, Y: batch_y})).sum(axis=0))

                Pre_y = np.array(sess.run(y_pre,feed_dict={X:x_test,Y:y_test}))

                for i in range(Pre_y.shape[0]):
                    for j in range(Pre_y.shape[1]):
                        if Pre_y[i][j] > 0.5 or Pre_y[i][j] == 0.5:
                            Pre_y[i][j]=1
                        else:
                            Pre_y[i][j]=0
                a = Pre_y -y_test

                q, w = Pre_y.shape[0], Pre_y.shape[1]
                count = 0
                for i in range(q):
                    for j in range(w):
                        if a[i, j] == 1:
                            count += 1
                print('Weight_loss ',count)

                Hamming_loss = hamming_loss(Pre_y,y_test)
                Jaccard_loss = jaccard_similarity_score(y_test, Pre_y)
                f1_macro = f1_score(y_test, Pre_y, average='macro')
                print('\nHamming_loss: {}  Jaccard_loss: {}  f1_micro:{}'.format(Hamming_loss,Jaccard_loss,f1_macro))


C2AE = C2AE()
C2AE.fit()
