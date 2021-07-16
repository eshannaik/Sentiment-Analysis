import flask
from flask import render_template,jsonify,request
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import h5py
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer
import re

token = ToktokTokenizer()

def clean_text(text):
	text = text.lower()
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "can not ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"\'scuse", " excuse ", text)
	text = re.sub(r"\'\n", " ", text)
	text = re.sub(r"\'\xa0", " ", text)
	text = re.sub(r'\s+', ' ', text)
	text = text.strip(' ')
	return text

nltk.download('stopwords')
nltk.download('wordnet')

lemma=WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lemitizeWords(text):
	words=token.tokenize(text)
	listLemma=[]
	for w in words:
		x=lemma.lemmatize(w, pos="v")
		listLemma.append(x)
	return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
	stop_words = set(stopwords.words("english"))
	words=token.tokenize(text)
	filtered = [w for w in words if not w in stop_words]
	return ' '.join(map(str, filtered))

punct = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

def strip_list_noempty(mylist):
	newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
	return [item for item in newlist if item != '']

def clean_punct(text): 
	words=token.tokenize(text)
	punctuation_filtered = []
	regex = re.compile('[%s]' % re.escape(punct))
	remove_punctuation = str.maketrans(' ', ' ', punct)
	for w in words:
			punctuation_filtered.append(regex.sub('', w))
	filtered_list = strip_list_noempty(punctuation_filtered)
	return ' '.join(map(str, filtered_list))

def emoji(text):
  emoji_pattern = re.compile("["
                               u"\xF0\x9F\x98\x81\xF0\x9F\x98\x81-\xF0\x9F\x98\x81\xF0\x9F\x98\x89"  # emoticons
                               u"\xF0\x9F\x98\x8A-\xF0\x9F\x98\x8F"
                               u"\xF0\x9F\x98\x81\xF0\x9F\x98\x91-\xF0\x9F\x98\x81\xF0\x9F\x98\x99"
                               u"\xF0\x9F\x98\x8A-\xF0\x9F\x98\x8F"
                               u"\x80"
                               "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'',text)

def predict(i):
	test_samples = [i]
	test_samples = clean_text(i)
	test_samples = lemitizeWords(i)
	test_samples = stopWordsRemove(i)
	test_samples = clean_punct(i)
	test_samples = emoji(i)

	model = load_model('./sentiment_analysis.h5')
	print(" <-----------------------------------------------Loaded model from disk--------------------------------------------> ")

	test_samples = [test_samples]

	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(analyzer='word', dtype='uint8')
	x1 = vectorizer.fit_transform(test_samples).toarray()

	sentiment = model.predict(x1)

	if sentiment[0] > 0.5:
		sentiment_str = "The given tweet is a racist/sexist tweet"
	else:
		sentiment_str = "The given tweet is not a racist/sexist tweet"

	return sentiment_str,sentiment

app = flask.Flask(__name__)

@app.route('/')
def my_form():
	return render_template('index.html')

@app.route('/prediction',methods=["POST","GET"])
def prediction():
	if request.method=="POST":
		msg = request.form['Tweet']
		response,s = predict(msg)

		return render_template('result.html',final_result=response,message = msg,score = s[0])
	return None


if __name__ == "__main__":
	app.run()
	