from collections import defaultdict  # default-set
import math


class TestHMM():
    def __init__(self):
        self.tran = defaultdict(lambda: 0)
        self.emit = defaultdict(lambda: 0)
        self.poss_tags = defaultdict(lambda: 0)
        self.lambda_1 = 0.95
        self.V = 1000000

    def load(self, model_file_name):
        with open(model_file_name, 'r') as model_file:
            data = model_file.readlines()
        for line in data:
            tag_type, context, word, prob = line.strip().split(" ")
            self.poss_tags[context] = 1
            context_word = context + " " + word
            if tag_type == "T":
                self.tran[context_word] = float(prob)
            else:
                self.emit[context_word] = float(prob)

    def prob_tran(self, next, prev):
        return self.tran[prev + " " + next]

    def prob_emit(self, word, next):
        next_word = next + " " + word
        return self.lambda_1 * self.emit[next_word] + (1 - self.lambda_1) / self.V

    def forward_calc_main(self, words, best_score, best_edge):
        for i in range(len(words)):
            for prev in self.poss_tags.keys():
                for next in self.poss_tags.keys():
                    prev_tag = str(i) + " " + prev
                    next_tag = str(int(i+1)) + " " + next
                    if prev_tag in best_score.keys() and (prev + " " + next) in self.tran.keys():
                        score = best_score[prev_tag] - \
                            math.log(self.prob_tran(next, prev)) - \
                            math.log(self.prob_emit(words[i], next))
                        if next_tag not in best_score.keys() or best_score[next_tag] > score:
                            best_score[next_tag] = score
                            best_edge[next_tag] = prev_tag
        return best_score, best_edge

    def forward_calc_last(self, l, best_score, best_edge):
        for tag in self.poss_tags.keys():
            prev_tag = str(l) + " " + tag
            next_tag = str(int(l+1)) + " </s>"
            if prev_tag in best_score.keys() and (tag + " </s>") in self.tran.keys():
                score = best_score[prev_tag] + - \
                    math.log(self.prob_tran("</s>", tag))
                if next_tag not in best_score.keys() or best_score[next_tag] > score:
                    best_score[next_tag] = score
                    best_edge[next_tag] = prev_tag
        return best_score, best_edge

    def forward_step(self, line):
        words = line.strip().split()
        l = len(words)
        best_score = defaultdict(lambda: 10**10)
        best_edge = defaultdict(str)
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None
        best_score, best_edge = self.forward_calc_main(
            words, best_score, best_edge)
        best_score, best_edge = self.forward_calc_last(
            l, best_score, best_edge)
        return best_edge

    def back_step(self, line, best_edge):
        words = line.strip().split()
        l = len(words)
        tags = []
        next_edge = best_edge[str(int(l+1)) + " </s>"]
        while next_edge != "0 <s>":
            position, tag = next_edge.split(" ")
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        return ' '.join(tags)

    def pos_tagging(self, input_file_name, output_file_name):
        with open(input_file_name, 'r') as input_file:
            data = input_file.readlines()
        with open(output_file_name, 'w') as output_file:
            for line in data:
                best_edge = self.forward_step(line)
                trg_str = self.back_step(line, best_edge)
                output_file.write(f"{trg_str}\n")


if __name__ == "__main__":
    # model_file = "04-train-output.txt"
    model_file = "04-model.txt"
    # input_file = "test/05-test-input.txt"
    input_file = "data/wiki-en-test.norm"
    # output_file = "04-test-output.txt"
    output_file = "my_answer.pos"

    test_hmm = TestHMM()
    test_hmm.load(model_file)
    test_hmm.pos_tagging(input_file, output_file)
