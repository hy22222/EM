import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm
import numpy as np

#1词典
class Dictionary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
#2文本序列化
class Corpus(object):

    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # step 1
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # step 2
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # step 3
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

#3模型
class LSTMmodel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.dropout(out)
        out = self.linear(out)
        return out, (h, c)
class Net2(nn.Module):
    def __init__(self, vocab_size, embed_size,  hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)  #
        # GRU
        self.gru = nn.GRU(embed_size, hidden_size,  batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.01)
    def forward(self,  x):
        x = self.embedding(x)
        out, hn = self.gru(x)
        x = out.reshape(out.size(0) * out.size(1), out.size(2))
        x = self.dropout(x)
        outs = self.fc(x)
        return outs
#4训练函数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 10
batch_size = 50
seq_length = 30
learning_rate = 0.001

corpus = Corpus()
ids = corpus.get_data('sdyx.txt', batch_size)
vocab_size = len(corpus.dictionary)
model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)
#GRU
# model=Net2(vocab_size,embed_size, hidden_size)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):

    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    loss_list=[]
    for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        states = [state.detach() for state in states]
        outputs, states = model(inputs, states)
        # outputs = model(inputs)
        loss = cost(outputs, targets.reshape(-1))
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        loss_list.append(loss.item())
    print("epocth={}\tloss={}".format(epoch, np.mean(loss_list)))
#5生成文章
num_samples = 300

article = str()

state = (torch.zeros(num_layers, 1, hidden_size).to(device),
        torch.zeros(num_layers, 1, hidden_size).to(device))

prob = torch.ones(vocab_size)
_input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
for i in range(num_samples):
    output, state = model(_input, state)
    # output = model(_input)
    prob = output.exp()
    word_id = torch.multinomial(prob, num_samples=1).item()

    _input.fill_(word_id)

    word = corpus.dictionary.idx2word[word_id]
    word = '\n' if word == '<eos>' else word
    article += word
print(article)

with open("predict.txt", "w") as f:
    f.write(article)  # 自带文件关闭功能，不需要再写f.close()