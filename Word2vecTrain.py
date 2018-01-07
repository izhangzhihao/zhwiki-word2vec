import os, sys
import multiprocessing
import gensim


def word2vec_train(input_file, output_file):
    sentences = gensim.models.word2vec.LineSentence(input_file)
    model = gensim.models.Word2Vec(sentences, size=300, min_count=5, sg=1, workers=multiprocessing.cpu_count())
    model.save(output_file)
    # model.save_word2vec_format(output_file + '.vector', binary=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python Word2vecTrain.py infile outfile")
        sys.exit()
    input_file, output_file = sys.argv[1], sys.argv[2]
    word2vec_train(input_file, output_file)