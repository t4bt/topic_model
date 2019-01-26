from __future__ import print_function

cimport cython
cimport numpy as np
import numpy as np
import pandas as pd
import random
from scipy.special import digamma
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option("display.width", 200)


def sampling(np.ndarray[double, ndim=1] prob):
    # prob = [0.3, 0.1, 0.2, 0.1, 0.2, 0.1]
    # probから確率的にインデックス(トピックk)を返す
    cdef int i
    cdef double z=0.0
    cdef double remaining
    for i in range(len(prob)):
        z += prob[i]
    remaining = random.uniform(0, z)
    for i in range(len(prob)):
        remaining -= prob[i]
        if remaining <= 0:
            return i


class LDA:

    def __init__(self, corpus, id2word, int num_topic=10, int iterations=100):
        cdef int it, d, n, k, v
        self.corpus = corpus
        self.id2word = id2word

        self.K = num_topic      # トピック数
        self.D = len(corpus)    # 文書数
        self.V = len(id2word)   # 単語数

        # alpha
        cdef np.ndarray[double, ndim=1] alpha \
                = np.ones(self.K, dtype=np.double) / self.K
        # beta
        cdef np.ndarray[double, ndim=1] beta \
                = np.ones(self.V, dtype=np.double) * 0.3

        # 文書dでトピックkが割り当てられた単語数（文書×トピック）
        cdef np.ndarray[double, ndim=2] N_dk \
                = np.zeros((self.D, self.K), dtype=np.double)
        # 文書集合全体で語彙vにトピックkが割り当てられた単語数（トピック×単語）
        cdef np.ndarray[double, ndim=2] N_kv \
                = np.zeros((self.K, self.V), dtype=np.double)
        # 文書集合全体でトピックkが割り当てられた単語数
        cdef np.ndarray[double, ndim=1] N_k \
                = np.zeros(self.K, dtype=np.double)
        # 文書dに含まれる単語数(文書長)
        nd = [float(sum([word[1] for word in doc])) for doc in corpus]
        cdef np.ndarray[double, ndim=1] N_d \
                = np.array(nd, dtype=np.double)
        # 文書dのn番目の単語のトピック
        cdef np.ndarray[int, ndim=3] z_dn \
                = np.zeros((self.D, int(max(N_d)), 2), dtype=np.int32)

        for d in range(self.D):
            for n in range(len(corpus[d])):
                z_dn[d,n,0] = corpus[d][n][0]
                z_dn[d,n,1] = -1

        # トピック分布theta
        cdef np.ndarray[double, ndim=2] theta \
                = np.zeros((self.D, self.K), dtype=np.double)
        # 単語分布phi
        cdef np.ndarray[double, ndim=2] phi \
                = np.zeros((self.K, self.V), dtype=np.double)

        # トピックのサンプリング確率
        cdef np.ndarray[double, ndim=1] p \
                = np.zeros(self.K, dtype=np.double)

        for it in range(iterations):
            print("\r{}/{}".format(it+1,iterations), end="")
            for d in range(self.D):
                for n in range(len(corpus[d])):
                    w = z_dn[d,n,0]     # 文書dのn番目の単語ID
                    z = z_dn[d,n,1]     # 文書dのn番目の単語のトピック

                    # カウントからz_dnの割り当て分を引く ######################
                    #   文書dのn番目のトピックをサンプリングする確率は、
                    #   そのトピックを除いたトピック集合Z_\dnと、
                    #   文書集合Wが与えられた時のトピックz_dnの条件付確率
                    if z > -1:
                        N_dk[d,z] -= 1
                        N_kv[z,w] -= 1
                        N_k[z] -= 1

                    # サンプリング確率を計算 ##################################
                    p = (N_dk[d,:]+alpha) * (N_kv[:,w]+beta[w]).T \
                            / (N_k+np.sum(beta))

                    # トピックをサンプリング ##################################
                    z = sampling(p)
                    z_dn[d,n,1] = z

                    # カウントに新たに割り当てたトピックの分を加える ##########
                    N_dk[d,z] += 1
                    N_kv[z,w] += 1
                    N_k[z] += 1

            # ハイパーパラメータを更新 ########################################
            alpha *= (np.sum(digamma(N_dk + alpha), axis=0) \
                        - self.D * digamma(alpha)) \
                    / (np.sum(digamma(N_d + np.sum(alpha))) \
                        - self.D * digamma(np.sum(alpha)))
            beta *= (np.sum(digamma(N_kv + beta), axis=0) \
                        - self.K * digamma(beta)) \
                    / (np.sum(digamma(N_k + np.sum(beta))) \
                        - self.K * digamma(np.sum(beta)))
            print("alpha:{}, beta:{}".format(np.mean(alpha), np.mean(beta)))
        # alpha,betaからθ_dkとΦ_kvを算出
        for d in range(self.D):
            for k in range(self.K):
                theta[d,k] = (N_dk[d,k] + alpha[k]) \
                                    / (N_d[d] + np.sum(alpha))
        for k in range(self.K):
            for v in range(self.V):
                phi[k,v] = (N_kv[k,v] + beta[v]) \
                                    / (N_k[k] + np.sum(beta))
        self.phi = phi
        print("")

    def predict(self, corpus):
        topics = []
        for text in corpus:
            t = []
            for word in text:
                if word[0] >= self.V:
                    continue
                t.append(self.phi[:,word[0]] * word[1])
            t = np.sum(np.array(t), axis=0)
            t = t.flatten() / sum(t)
            topics.append(t)
        return np.array(topics)

    def show_topics(self, topn=10):
        Top_words = {}
        for k in range(self.K):
            phi_k = self.phi[k,:]
            top_nw = np.argsort(phi_k)[::-1][:topn]
            top_nw = [self.id2word[i] for i in top_nw]
            Top_words.update({'topic_{}'.format(k) : top_nw})
        data = pd.DataFrame(Top_words)
        print(data.T)
        return data.T
