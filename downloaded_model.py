import os, sys
import multiprocessing
import gensim


def word2vec_train(input_file, output_file):
    model = gensim.models.KeyedVectors.load('/Users/zhazhang/Downloads/word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.model')
    sentences = gensim.models.word2vec.LineSentence(input_file)
    model = gensim.models.Word2Vec(sentences, window=5, min_count=5,size=64, workers=4)
    model.save(output_file)
    # model.save_word2vec_format(output_file + '.vector', binary=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python Word2vecTrain.py infile outfile")
        sys.exit()
    input_file, output_file = sys.argv[1], sys.argv[2]
    word2vec_train(input_file, output_file)