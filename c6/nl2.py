import gensim.models
from gensim import utils
import pprint
from gensim.test.utils import common_texts
from gensim.test.utils import datapath

print(common_texts)
sentences = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

class Corpus:
    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            result = utils.simple_preprocess(line)
            print(result)
            yield result

# sentences = common_texts
# sentences = list(map(lambda x: utils.simple_preprocess(x), sentences))

sentences = Corpus()

# print(sentences)

model = gensim.models.Word2Vec(sentences=sentences)

# pprint.pprint(len(model.wv))
wv = model.wv
print(wv['paris'] - wv['france'] + wv['afghanistan'])
print(wv['kabul'])

