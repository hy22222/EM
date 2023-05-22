# import os
# import re
# import jieba
# import math
# import torch
# from rouge import Rouge
#
# import evaluate
# # #读stopwords
# f = open('cn_stopwords.txt','r',encoding='utf-8')
# stopwords = f.read().splitlines()
# # 去除符号及中文无意义stopwords
# def filter(text):
#     pattern = '|'.join(stopwords)
#     c = re.sub(pattern, '', text)
#     d = re.sub(r'\*', '', c)
#     e = re.sub(r'\.', '',d)
#     f = re.sub(r'\n', '', e)
#     g = re.sub(r'\u3000', '', f)
#     return g
# with open("actual.txt", "r", encoding="utf-8") as f:  # 打开文件
#     data1 = f.read()  # 读取文件
# with open("predict.txt", "r", encoding="utf-8") as f:  # 打开文件
#     data2 = f.read()  # 读取文件
# pred = data1
# ideal = data2
# # actual_data = filter(data1)
# # predict_data = filter(data2)
# # pred = predict_data
# # ideal = actual_data
#
# # Rouge()按空格分隔gram，所以要在中文的字和字之间加上空格
# pred, ideal = ' '.join(pred), ' '.join(ideal)
#
# # 计算字粒度的rouge-1、rouge-2、rouge-L
# rouge = Rouge()
# rouge_scores = rouge.get_scores(hyps=pred, refs=ideal)
# print(rouge_scores[0]["rouge-1"])
# print(rouge_scores[0]["rouge-2"])
# print(rouge_scores[0]["rouge-l"])

import jieba
import math


class LanguageModel:
    def __init__(self, corpus_file, n=2):
        self.n = n
        self.counts = {}
        self.totals = {}
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = list(jieba.cut(line.strip()))
                for i in range(len(words) - n + 1):
                    ngram = ''.join(words[i:i + n])
                    if ngram not in self.counts:
                        self.counts[ngram] = 0
                        self.totals[ngram[:-1]] = 0
                    self.counts[ngram] += 1
                    self.totals[ngram[:-1]] += 1

    def probability(self, ngram):
        if ngram[:-1] in self.totals:
            return self.counts.get(ngram, 0) / self.totals[ngram[:-1]]
        else:
            return 0.0

    def perplexity(self, test_file):
        test_ngrams = 0
        test_likelihood = 0.0
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = list(jieba.cut(line.strip()))
                for i in range(self.n - 1, len(words)):
                    ngram = ''.join(words[i - self.n + 1:i + 1])
                    test_ngrams += 1
                    test_likelihood += math.log2(self.probability(ngram) + 1e-100)
        test_perplexity = 2 ** (-test_likelihood / test_ngrams)
        return test_perplexity


# Usage:
lm = LanguageModel('actual.txt', n=2)
test_ppl = lm.perplexity('predict.txt')
print('Test Perplexity: %.3f' % test_ppl)



