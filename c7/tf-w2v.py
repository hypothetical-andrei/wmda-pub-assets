sentence = "this is a sentence and it is a good sentence"

tokens = list(sentence.lower().split())

print(tokens)

vocab = {}
index = 1

vocab['<pad>'] = 0
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1

# print(vocab)

inverse_vocab = {index : token for token, index in vocab.items()}

# print(inverse_vocab)

example_sequence = [vocab[word] for word in tokens]

print(example_sequence)