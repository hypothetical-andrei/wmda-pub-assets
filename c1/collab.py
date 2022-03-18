import data_sample
import distances
import loaders
import transform
import json
from json.decoder import JSONDecodeError

def top_matches(prefs, me, n = 5, similarity = distances.euclidean):
  scores = [(similarity(prefs, me, other), other) for other in prefs if other != me]
  scores.sort()
  scores.reverse()
  return scores[:n]

def get_recommendations(prefs, me, similarity = distances.euclidean, n = 5):
  totals = {}
  similarity_sums = {}
  for other in prefs:
    if other == me:
      continue
    sim = similarity(prefs, me, other)
    if sim == 0:
      continue
    for item in prefs[other]:
      if item not in prefs[me]:
        totals.setdefault(item, 0)
        totals[item] = totals[item] + prefs[other][item] * sim
        similarity_sums.setdefault(item, 0)
        similarity_sums[item] = similarity_sums[item] + sim
  rankings = [(total / similarity_sums[item], item) for item, total in totals.items()]
  rankings.sort()
  rankings.reverse()
  return rankings[:5]

def get_similar_items(prefs, n = 5, similarity = distances.euclidean):
  result = {}
  count = len(prefs)
  n = 0
  for item in prefs:
    n = n + 1
    if n % 100 == 0:
      print('processed {n} / {count}'.format(n = n, count = count))
    scores = top_matches(prefs, item, similarity = similarity)
    result[item] = scores
  return result

def get_recommended_items(prefs, item_similarities, me):
  my_ratings = prefs[me]
  scores = {}
  totals = {}
  for item, rating in my_ratings.items():
    for sim, other_item in item_similarities[item]:
      if other_item in my_ratings:
        continue
      scores.setdefault(other_item, 0)
      scores[other_item] = scores[other_item] + sim * rating
      totals.setdefault(other_item, 0)
      totals[other_item] = totals[other_item] + sim
  rankings = [(score / totals[item], item) for item, score in scores.items()]
  rankings.sort()
  rankings.reverse()
  return rankings

def main():
  # data = data_sample.critics
  data = loaders.load_movie_lens()
  transformed_data = transform.tranform_prefs(data)
  # print(transformed_data)
  # print(top_matches(data, 'Lisa Rose'))
  # top_recommendations = get_recommendations(data, 'Toby')
  # top_recommendations = get_recommendations(data, '3')
  # print(top_recommendations)
  # print(top_matches(transformed_data, 'Lady in the Water'))
  # item_similarities = get_similar_items(transformed_data)
  try:
    with open('items.json', encoding='utf-8') as f:
      item_similarities = json.loads(f.read())
  except (FileNotFoundError, JSONDecodeError):
    item_similarities = get_similar_items(transformed_data)
    with open('items.json', 'w', encoding='utf-8') as f:
      f.write(json.dumps(item_similarities))
  print(get_recommended_items(data, item_similarities, '3'))

if __name__ == '__main__':
  main()