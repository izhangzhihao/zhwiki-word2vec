import os, sys
import multiprocessing
import gensim


def word2vec_train(input_file, output_file, word2vec_format_file):
    sentences = gensim.models.word2vec.LineSentence(input_file)
    model = gensim.models.Word2Vec(sentences, size=256, min_count=5, sg=1, workers=multiprocessing.cpu_count(),iter=10)
    model.save(output_file)
    model.wv.save_word2vec_format(word2vec_format_file, binary=False)
    # model.save_word2vec_format(output_file + '.vector', binary=True)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python Word2vecTrain.py infile model_file word2vec_format_file")
        sys.exit()
    input_file, output_file,word2vec_format_file = sys.argv[1], sys.argv[2], sys.argv[3]
    word2vec_train(input_file, output_file, word2vec_format_file)