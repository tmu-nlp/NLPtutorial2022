import argparse
from collections import defaultdict

class Trainhmm:
    def __init__(self):  # class initialization
        self.emit = defaultdict(lambda: 0)
        self.transition = defaultdict(lambda: 0)
        self.context = defaultdict(lambda: 0)
    
    def prepare(self, read_file):  # prepare some dictionaries
        with open(read_file, "r") as read: 
            for line in read:
                previous = "<s>"  # first tag
                self.context["<s>"] += 1
                wordtags = line.split()
                for wordtag in wordtags:
                    word = wordtag.split("_")[0]
                    tag = wordtag.split("_")[1]
                    self.transition[f'{previous} {tag}'] += 1
                    previous = tag  # update tag
                    self.context[tag] += 1
                    self.emit[f'{tag} {word}'] += 1
                self.transition[f'{previous} </s>'] += 1
    
    
    def prob_cal(self, out_file):  # calculate transition-probability and emission-probability
        with open(out_file, "w") as out:  # calculate transition-probability
            for key, value in sorted(self.transition.items()):
                previous_tag = key.split()[0]
                next_tag = key.split()[1]
                t = value / self.context[previous_tag]  # t = transition-probability
                out.write("T " + key + ' {:.6f}'.format(t))
                out.write("\n")

            for key, value in sorted(self.emit.items()):  # caluculate emission-probability
                tag = key.split()[0]
                word = key.split()[1]
                t = value / self.context[tag]  # t = emission probability
                out.write("E " + key + ' {:.6f}'.format(t))
                out.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('part-of-speech estimation(train)')  # make parse and describe about this program

    parser.add_argument('file1', help = 'train_file_name')  # add argument and explain this argument
    parser.add_argument('file2', help = 'out_file_name')  # same as above

    args = parser.parse_args()  # analyze argument

    train_data = args.file1
    out_file = args.file2

    print('train_data = ' + args.file1)  # output character of argument
    print('out_file = ' + args.file2)  # same as above

    t = Trainhmm()
    t.prepare(train_data)
    t.prob_cal(out_file)

"""
if input 'python3 train-hmm.py wiki-en-train.norm_pos train-result.txt' in command line,

'''
train_data = wiki-en-train.norm_pos
out_file = train-result.txt
'''
"""



"""
if input 'python3 train-hmm.py -h'

'''
usage: part-of-speech estimation(train) [-h] file1 file2

positional arguments:
  file1       train_file_name
  file2       out_file_name

optional arguments:
  -h, --help  show this help message and exit
'''
"""