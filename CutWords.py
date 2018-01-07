import sys
import io
import jieba

jieba.load_userdict("dict.txt")

def cut_words(input_file, output_file):
    count = 0
    with io.open(output_file, mode = 'w', encoding = 'utf-8') as outfile:
        with io.open(input_file, mode = 'r', encoding = 'utf-8') as infile:
            for lines in infile:
                line = lines.strip()
                if line.startswith('doc'): # start or end of a passage
                    if line == 'doc': # end of a passage
                        outfile.write('\n')
                        count = count + 1
                        if(count % 1000 == 0):
                            print('%s articles were finished.......' %count)
                    continue
                for word in jieba.cut(line):
                    outfile.write(word + ' ')
    print('%s articles were finished.......' %count)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python CutWords.py input_file output_file")
        sys.exit()
    input_file, output_file = sys.argv[1], sys.argv[2]
    cut_words(input_file, output_file)