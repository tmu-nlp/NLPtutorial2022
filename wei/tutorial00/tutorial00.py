from collections import defaultdict


def count_words(input_file, output_file):
    word_cnts = defaultdict(lambda :0)
    with open(input_file,'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            words = line.strip().split()
            for word in words:
                if word in word_cnts:
                    word_cnts[word] += 1
                else:
                    word_cnts[word] = 1
        f_out.write(f'{word_cnts.items()}\n')
    # keyによってソート
    print('\n'.join([f'{w}\t{c}' for w,c in sorted(word_cnts.items())]))

    print(f'vocab is {len(word_cnts)}')   #vocab is 5233


if __name__ == '__main__':
    train_file = '../data/wiki-en-train.word'
    res = 'results.txt'
    count_words(train_file, res)

