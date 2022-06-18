'''
train_file: ../data/wiki-en-train.word
'''

from collections import defaultdict


def unigram(input_file, model_output):

    w_cnts = defaultdict(int)
    total_cnts = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            words = line.strip().split()
            words.append('</s>')
            for word in words:
                w_cnts[word] += 1
                total_cnts += 1
        print(f'vocab size is {len(w_cnts)}')   #vocab size is 5234


    with open(model_output, 'w', encoding='utf-8') as f_out:
        for w, c in w_cnts.items():
            prob = c/total_cnts
            f_out.write(f'{w}\t{prob:.7f}\n')



if __name__ == '__main__':
    train_file = '../data/wiki-en-train.word'
    model_res = 'model_file.txt'
    unigram(train_file, model_res)

