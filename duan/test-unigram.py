import sys
import math
import argparse

def getArgs():
    parser = argparse.ArgumentParser(description="test the language model")
    parser.add_argument(
        "-f", "--input",
        dest="test_file",
        type=argparse.FileType("r"),
        required=True,
    )

    parser.add_argument(
        "-l", "--lm",
        dest="lm_file",
        type=argparse.FileType("r"),
        required=True,
    )

    parser.add_argument(
        "-o", "--output",
        dest="eval_file",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )

    return parser.parse_args()

def get_pml():
    pml = dict()
    for line in args.lm_file:
        word = line.strip().split("\t")[0]
        prob = float(line.strip().split("\t")[1])
        pml[word] = prob
    return pml

def calc_eval(pml, lam, N):
    words_count = .0
    unk_count = .0
    H = .0
    p = dict()
    for line in args.test_file:
        words = line.strip().split()
        words.append("</s>")
        for word in words:
            p[word] = lam * pml.get(word, 0) + (1-lam)/N
            H -= math.log(p[word], 2)
            if word not in pml:
                unk_count += 1
            words_count += 1

    H = H/words_count
    coverage = (words_count-unk_count)/words_count
    perplexity = 2**H
    return H, perplexity, coverage

def main():
    N = 10**6
    lam_one = 0.95
    pml = get_pml()
    H, perplexity, coverage = calc_eval(pml, lam_one, N)
    print >> args.eval_file, "entropy : %f" % H
    print >> args.eval_file, "perplexity : %f" % perplexity
    print >> args.eval_file, "coverage : %f" % coverage

if __name__ == '__main__':
    args = getArgs()
    main()