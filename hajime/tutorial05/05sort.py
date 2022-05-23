from pprint import pprint

model_dict = dict()
# model_file = "output/05/1gram/model-05-100.txt"
# model_file = "output/05/2gram/model-05-2gram-100.txt"
model_file = "output/05/prep/model-05-prep-100.txt"

with open(model_file, 'r') as m_file:
    for line in m_file:
        key, value = line.strip().split(" ")
        model_dict[key] = int(value)

sort_model = sorted(model_dict.items(), key=lambda x: x[1])

pprint(sort_model)
