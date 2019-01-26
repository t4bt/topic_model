from gensim import corpora


def load_wakati_docs(filename):
    # 1行1文書の分かち書き済みテキストファイル
    texts = []
    for line in open(filename, 'r'):
        texts.append(line.split(' '))
    return texts

def create_dict_and_corpus(texts, no_below=10, no_above=0.2):
    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.save_as_text('./docs.dic')
    # コーパス作成
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('./corpus.mm', corpus)
    return dictionary, corpus


if __name__ == "__main__":
    import sys
    from lda import LDA
    import time
    texts = load_wakati_docs(sys.argv[1])
    dictionary, corpus = create_dict_and_corpus(texts)
    t0 = time.time()
    lda = LDA(corpus, dictionary, num_topic=10, iterations=1000)
    t1 = time.time()
    lda.show_topics()
    print("Elapsed time: {}".format(t1-t0))
