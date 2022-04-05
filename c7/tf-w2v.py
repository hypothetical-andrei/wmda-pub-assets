import tensorflow as tf
from tensorflow.keras import layers
import re
import string
import tqdm

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

# print(text_ds)

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=13,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# with open(path_to_file) as f :
#   lines = f.read().splitlines()

# print(lines[:10])

text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

vocab_size = 4096
sequence_length = 10

vectorize_layer = layers.TextVectorization(
  standardize = custom_standardization,
  max_tokens = vocab_size, 
  output_sequence_length = sequence_length,
  output_mode = 'int'
)

vectorize_layer.adapt(text_ds.batch(1024))

inverse_vocab = vectorize_layer.get_vocabulary()

# print(inverse_vocab[:10])

text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()

# print(text_vector_ds)

sequences = list(text_vector_ds.as_numpy_iterator())

# print(len(sequences))

targets, contexts, labels = generate_training_data(
  sequences = sequences, 
  window_size = 2, 
  num_ns = 4, 
  vocab_size = vocab_size,
  seed = 13
)


BUFFER_SIZE = 10000

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(1024, drop_remainder=True)

# print(dataset)
embedding_dim = 128

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size, embedding_dim, input_length = 1, name="word2vec_embedding")
    self.context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length = num_ns + 1)
  def call(self, pair):
    target, context = pair
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis = 1)
    word_embedding = self.target_embedding(target)
    context_embedding = self.context_embedding(context)
    dots = tf.einsum('be,bce->bc', word_embedding, context_embedding)
    return dots

word2vec = Word2Vec(vocab_size, embedding_dim)

word2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

word2vec.fit(dataset, epochs = 10)

# weights = word2vec.get_layer('word2vec_embedding').get_weights()[0]
# vocab = vectorize_layer.get_vocabulary()

# out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
# out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

# for index, word in enumerate(vocab):
#   if index == 0:
#     continue  # skip 0, it's padding.
#   vec = weights[index]
#   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
#   out_m.write(word + "\n")
# out_v.close()
# out_m.close()