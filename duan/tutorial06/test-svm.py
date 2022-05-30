# python3 ./nlptutorial/script/grade-prediction.py ./nlptutorial/data/titles-en-test.labeled ./NLPtutorial2022/duan/tutorial06/test.txt
# Accuracy = 92.454835%

from collections import defaultdict

def create_map(filename:str):
    weight_map = defaultdict(lambda:0)
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            word = line[0]
            weight = float(line[1])
            weight_map[word] = weight
    return weight_map

def test():
    weight_map = create_map('./NLPtutorial2022/duan/tutorial06/model06.txt')
    t = ''
    with open('./nlptutorial/data/titles-en-test.word') as f:
        for line in f:
            words = line.strip().split(' ')
            prob = 0
            for word in words:
                prob += weight_map[word]
            if prob >= 0:
                y = 1
            else:
                y = -1
            t += f'{y}\n'
    with open('./NLPtutorial2022/duan/tutorial06/test.txt','w') as f:
        f.write(t)

if __name__ == '__main__':
    test()