import sys
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description="train the unigram language model")

    parser.add_argument(
        "-f", "--input",
        dest="train_file",
        type=argparse.FileType("r"),
        required=True,
    )

    parser.add_argument(
        "-o", "--output",
        dest="model_file",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )

    return parser.parse_args()

def count_words():
    all_count = 0
    words_count = dict()
    for line in args.train_file:
        words = line.strip().split()
        words.append("</s>")
        for word in words:
            words_count[word] = words_count.get(word, 0) + 1
            all_count += 1

    return all_count, words_count

def output_lm(all_count, words_count):
    for word, count in sorted(words_count.items()):
        print >> args.model_file, "%s\t%f" % (word, float(count)/all_count)

def main():
    all_count, words_count = count_words()
    output_lm(all_count, words_count)

if __name__ == '__main__':
    args = getArgs()
    main()