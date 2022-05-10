from collections import defaultdict  # default-set


class TrainHMM():
    def __init__(self):
        self.emit = defaultdict(lambda: 0)
        self.tran = defaultdict(lambda: 0)
        self.cont = defaultdict(lambda: 0)
        self.lambda_1 = 0.95
        self.prev = ""
        self.sort_tran = []
        self.sort_emit = []

    def train(self, train_file_name):
        with open(train_file_name, 'r') as train_file:
            data = train_file.readlines()
        for line in data:
            self.prev = "<s>"
            self.cont[self.prev] += 1
            wordtags = line.strip().split(" ")
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                self.tran[self.prev + " " + tag] += 1
                self.cont[tag] += 1
                self.emit[tag + " " + word] += 1
                self.prev = tag
            self.tran[self.prev + " </s>"] += 1

    def sort_data(self):
        self.sort_tran = sorted(self.tran.items(), key=lambda x: x[0])
        self.sort_emit = sorted(self.emit.items(), key=lambda x: x[0])

    def prob_output(self, output_file_name):
        self.sort_data()
        with open(output_file_name, 'w') as output_file:
            for key, value in self.sort_tran:
                previous, word = key.split(" ")
                output_file.write(
                    f"T {key} {float(value) / self.cont[previous]}\n")
            for key, value in self.sort_emit:
                tag, word = key.split(" ")
                output_file.write(f"E {key} {float(value) / self.cont[tag]}\n")


if __name__ == "__main__":
    # trg_file = "./test/05-train-input.txt"
    trg_file = "./data/wiki-en-train.norm_pos"
    # out_file = "04-train-output.txt"
    out_file = "04-model.txt"

    train_hmm = TrainHMM()
    train_hmm.train(trg_file)
    train_hmm.prob_output(out_file)
