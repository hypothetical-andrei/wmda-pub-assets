import tensorflow as tf

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
vocabulary_size = len(vocab)

example_sequence = [vocab[word] for word in tokens]

# print(example_sequence)

window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
  example_sequence,
  vocabulary_size = vocabulary_size,
  window_size = window_size,
  negative_samples = 0
)

# print(len(positive_skip_grams))
# print(positive_skip_grams[:5])

# for target, context in positive_skip_grams[:5]:
#   print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")


target_word, context_word = positive_skip_grams[0]

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

num_ns = 4


negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
  true_classes = context_class,
  num_true = 1,
  num_sampled = num_ns,
  unique = True,
  range_max = vocabulary_size,
  name = "negative sampling",
  seed = 13
)

# print(negative_sampling_candidates)

# print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
context = tf.concat([context_class, negative_sampling_candidates], 0)

label = tf.constant([1] + [0] * num_ns, dtype="int64")

target = tf.squeeze(target_word)
context = tf.squeeze(context)
label = tf.squeeze(label)

# print(target)
# vocab_item = inverse_vocab[target.ref()]

# print(f"target {target} {inverse_vocab[target_word]}")
# print(context)
# print(label)

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

with open(path_to_file) as f :
  lines = f.read().splitlines()

print(lines[:10])