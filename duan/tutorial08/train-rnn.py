import random
import numpy as np
from collections import defaultdict

def train_rnn():
    hidden_layer, output_layer, x_ids, y_ids = init_network()
    vocab_size = len(x_ids)
    output_size = len(y_ids)
    with open('./nlptutorial/data/wiki-en-train.norm_pos') as f:
        f = list(f)
        for epoch in range(20):
            random.shuffle(f)
            for line in f:
                word_tag_list = line.strip().split()
                h_list = list()
                p_list = list()
                output_list = list()
                x_list = list()
                y_list = list()
                for t, word_tag in enumerate(word_tag_list):
                    word, tag = word_tag.split('_')
                    word = word.lower()
                    x_vector, y_vector = create_onehot(word, tag, x_ids, y_ids)
                    h_list, p_vector, output = forward_rnn(hidden_layer, output_layer, x_vector, h_list, t)
                    p_list.append(p_vector)
                    output_list.append(output)
                    x_list.append(x_vector)
                    y_list.append(y_vector)
                delta_w_rx, delta_w_rh, delta_b_r, delta_w_oh, delta_b_o = gradient_rnn(hidden_layer, output_layer, x_list, y_list, h_list, p_list, vocab_size, output_size)
                hidden_layer, output_layer = update_weight(hidden_layer, output_layer, delta_w_rx, delta_w_rh, delta_b_r, delta_w_oh, delta_b_o, lr=0.005)
    return hidden_layer, output_layer, x_ids, y_ids

def update_weight(hidden_layer, output_layer, delta_w_rx, delta_w_rh, delta_b_r, delta_w_oh, delta_b_o, lr):
    hidden_layer[0] += lr * delta_w_rx
    hidden_layer[1] += lr * delta_w_rh
    hidden_layer[2] += lr * delta_b_r
    output_layer[0] += lr * delta_w_oh
    output_layer[1] += lr * delta_b_o
    return hidden_layer, output_layer

def gradient_rnn(hidden_layer, output_layer, x_list, y_list, h_list, p_list, vocab_size, output_size, middle_node=8):
    l = len(y_list)
    delta_w_rx = np.zeros((vocab_size, middle_node))
    delta_w_rh = np.zeros((middle_node, middle_node))
    delta_b_r = np.zeros((middle_node, 1))
    delta_w_oh = np.zeros((middle_node, output_size))
    delta_b_o = np.zeros((output_size, 1))
    delta_r_ = np.zeros((middle_node, 1)) 
    for t in range(l-1, -1, -1):
        _delta_o = y_list[t] - p_list[t]        
        delta_w_oh += np.outer(h_list[t], _delta_o)  
        delta_b_o += _delta_o
        delta_r = np.dot(hidden_layer[1], delta_r_) + np.dot(output_layer[0], _delta_o)  
        delta_r_ = delta_r * (1-pow(h_list[t], 2)) 
        delta_w_rx += np.outer(x_list[t].T, delta_r_)  
        delta_b_r += delta_r_
        if t != 0:
            delta_w_rh += np.outer(h_list[t-1], delta_r_) 
    return delta_w_rx, delta_w_rh, delta_b_r, delta_w_oh, delta_b_o

def softmax(v):
    v = np.exp(v)
    v = v / np.sum(v)
    return v

def forward_rnn(h_layer, o_layer, x_vector, h_list, t):
    if t == 0:
        h_list.append(np.tanh(np.dot(h_layer[0].T, x_vector.T)+h_layer[2]))
    if t > 0:
        h_list.append(np.tanh(np.dot(h_layer[0].T, x_vector.T)+np.dot(h_layer[1].T, h_list[-1])+h_layer[2]))
    p_vector = softmax(np.dot(o_layer[0].T, h_list[-1])+o_layer[1])  
    output = np.argmax(p_vector)
    return h_list, p_vector, output
                
def create_onehot(word, tag, x_ids, y_ids):
    x_vector = np.zeros((1, len(x_ids)))
    y_vector = np.zeros((len(y_ids), 1))
    x_vector[0][x_ids[word]] = 1
    y_vector[y_ids[tag]][0] = 1
    return x_vector, y_vector

def init_network(middle_node=8):
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    with open('./nlptutorial/data/wiki-en-train.norm_pos') as f:
        for line in f:
            word_tag_list = line.strip().split()
            for word_tag in word_tag_list:
                word, tag = word_tag.split('_')
                word = word.lower()
                x_ids[word]
                y_ids[tag]
        vocab_size = len(x_ids)
        output_size = len(y_ids)
        hidden_layer = list()  
        output_layer = list() 
        hidden_layer.append(np.random.uniform(-0.1, 0.1, (vocab_size, middle_node)))  
        hidden_layer.append(np.random.uniform(-0.1, 0.1, (middle_node, middle_node))) 
        hidden_layer.append(np.zeros((middle_node, 1))) 
        output_layer.append(np.random.uniform(-0.1, 0.1, (middle_node, output_size)))  
        output_layer.append(np.zeros((output_size, 1))) 
        return hidden_layer, output_layer, x_ids, y_ids

if __name__ == '__main__':
    train_rnn()
