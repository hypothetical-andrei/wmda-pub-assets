
from collections import defaultdict

document = "Human machine interface for lab abc computer application computer"

text_corpus = [
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

stoplist = set('for a of the and to in'.split(' '))

texts = [[word for word in document.lower().split() if word not in stoplist] for document in text_corpus]

# print(texts)

frequency = defaultdict(int)
for text in texts:
  for token in text:
    frequency[token] += 1

# print(frequency)

processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
print(processed_corpus)

from gensim import corpora, models, similarities

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary.token2id)

new_doc = "Human computer interaction"
# print(dictionary.doc2bow(new_doc.lower().split()))


bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]



tfidf = models.TfidfModel(bow_corpus)

words = "graph trees".lower().split()
print(tfidf[dictionary.doc2bow(words)])


index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)


query_target = "system engineering".split()
query_bow = dictionary.doc2bow(query_target)
sims = index[tfidf[query_bow]]

print(sims)