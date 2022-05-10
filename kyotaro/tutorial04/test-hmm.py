import math
import argparse
from collections import defaultdict

class Testhmm:
    def __init__(self):  # class initalization
        self.transition = defaultdict(lambda: 0)
        self.emission = defaultdict(lambda: 0)
        self.possible_tags = defaultdict(lambda: 0)
        self.lmd = 0.95
        self.V = 1000000
    
    def loads_model(self, train):  # receive train-result and set some dictionaries
        for line in train:
            line = line.split()
            kind = line[0]
            context = line[1]
            word = line[2]
            prob = line[3]
            self.possible_tags[context] = 1
            if kind == "T":  # this means it is transition-probability
                self.transition[context + " " + word] = float(prob)
            else:  # this means it is emission-probability
                self.emission[context + " " + word] = float(prob)
        return
    
    def positive_step(self, line):  # front-step in Viterbi-algorithm
        words = line.strip().split()
        l = len(words)
        best_score = defaultdict(lambda: 0)
        best_edge = defaultdict(lambda: 0)
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None
        for i in range(l):
            for prev_tag in self.possible_tags.keys():
                for next_tag in self.possible_tags.keys():
                    p = (1 - self.lmd) / self.V
                    if f'{i} {prev_tag}' in best_score and self.transition[f'{prev_tag} {next_tag}']:
                        score = best_score[f'{i} {prev_tag}'] - math.log(self.transition[f'{prev_tag} {next_tag}'], 2) - math.log(p + self.lmd * self.emission[f'{next_tag} {words[i]}'], 2)
                        if not(f'{i + 1} {next_tag}' in best_score) or best_score[f'{i + 1} {next_tag}'] > score:
                            best_score[f'{i + 1} {next_tag}'] = score
                            best_edge[f'{i + 1} {next_tag}'] = f'{i} {prev_tag}'
            
            for last_tag in self.possible_tags.keys():  # add end symbol
                if f'{l} {last_tag}' in best_score and self.transition[f'{last_tag} </s>']:
                    score = best_score[f'{l} {last_tag}'] - math.log(self.transition[f'{last_tag} </s>'], 2)
                    if not(f'{l + 1} </s>' in best_score) or best_score[f'{l + 1} </s>'] > score:
                        best_score[f'{l + 1} </s>'] = score
                        best_edge[f'{l + 1} </s>'] = f'{l} {last_tag}'
        return best_edge
    
    def negative_step(self, line, best_edge):  # back-step in Viterbi-algorithm
        tags = []
        line = line.strip().split()
        l = len(line)
        next_edge = best_edge[f'{l + 1} </s>']
        while next_edge != "0 <s>":
            position = next_edge.split()[0]
            tag = next_edge.split()[1]
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        return tags
    
    def step_synthesis(self, test, out_file):  # positive-step combine with negative-step
        with open(out_file, "w") as out:     
            for line in test:
                best_edge = self.positive_step(line)
                route = " ".join(self.negative_step(line, best_edge))
                out.write(route)
                out.write("\n")
        return
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser('part-of-speech estimation(test)')
    parser.add_argument('file1', help = 'model_file_name')
    parser.add_argument('file2', help = 'test_file_name')
    parser.add_argument('file3', help = 'out_file_name')

    args = parser.parse_args()

    train_result = open(args.file1, "r").readlines()
    test_data = open(args.file2, "r").readlines()
    out_file = args.file3

    print('model_file = ' + args.file1)
    print('test_subject = ' + args.file2)
    print('out_file = ' + args.file3)
    
    t = Testhmm()
    t.loads_model(train_result)
    t.step_synthesis(test_data, out_file)


"""
if input 'python3 test-hmm.py train-result.txt wiki-en-test.norm my_answer.pos' in command line,

///
model_file = train-result.txt
test_subject = wiki-en-test.norm
out_file = my_answer.pos
///
"""



"""
if input 'python3 test-hmm.py -h' in command line,

///
usage: part-of-speech estimation(test) [-h] file1 file2 file3

positional arguments:
  file1       model_file_name
  file2       test_file_name
  file3       out_file_name

optional arguments:
  -h, --help  show this help message and exit
///
"""