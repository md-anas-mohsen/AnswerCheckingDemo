from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import json
import nltk
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')


def preprocess(sentence):
  return [w for w in sentence.lower().split() if w not in stop_words]


app = Flask(__name__)

ENV = 'dev'

if ENV == 'dev':
  app.debug = True
else:
  app.debug = False

model = pickle.load(open('physics_books_word2vec_model.pickle', 'rb'))


def checkAnswerWithWord2Vec(markingScheme, answer):
  minScore = float('inf')
  scoreSum = 0
  scores = []
  for sampleAnswer in markingScheme:
    distance = model.wmdistance(preprocess(sampleAnswer), preprocess(answer))
    scoreSum += distance
    minScore = min(distance, minScore)
    print(sampleAnswer)
    print('distance = %.4f' % distance)
    scores.append({"markSchemeAnswer": sampleAnswer, "distance": distance})
    print()
  average = scoreSum / len(markingScheme)
  print("average =  %.4f" % average)
  return {
    "minDistance": minScore,
    "averageDistance": average,
    "distances": scores
  }


@app.route('/api/word2vec-word-mover-distance-batch', methods=["POST"])
def word_mover_distance_batch():
  mark_scheme = request.form.get('mark_scheme')
  my_answer = request.form.get('my_answer')

  # print(request.get_json())
  # body = request.get_json()
  # mark_scheme = body['mark_scheme']
  # my_answer = body['my_answer']

  print(mark_scheme)
  print(my_answer)

  if mark_scheme is None:
    return json.dumps({
      "success": False,
      "message": "Please provide mark scheme as array"
    })

  if my_answer is None:
    return json.dumps({
      "success": False,
      "message": "Please provide my answer"
    })

  return json.dumps(checkAnswerWithWord2Vec(json.loads(mark_scheme),
                                            my_answer))


@app.route('/api/word2vec-word-mover-distance-single', methods=["POST"])
def word_mover_distance():
  gold_answer = request.form.get('gold_answer')
  my_answer = request.form.get('my_answer')

  # print(request.get_json())
  # body = request.get_json()
  # gold_answer = body['gold_answer']
  # my_answer = body['my_answer']

  print(gold_answer)
  print(my_answer)

  if gold_answer is None:
    return json.dumps({
      "success": False,
      "message": "Please provide gold answer"
    })

  if my_answer is None:
    return json.dumps({
      "success": False,
      "message": "Please provide my answer"
    })

  distance = model.wmdistance(preprocess(gold_answer), preprocess(my_answer))
  print('distance = %.4f' % distance)
  return json.dumps({"success": True, "distance": distance})


@app.route('/')
def index():
  return render_template('word2vec.html', max_condition=52)


@app.route('/word2vec/batch')
def word2VecBatch():
  return render_template('word2vecBatch.html', max_condition=52)


app.run(host='0.0.0.0', port=81)
