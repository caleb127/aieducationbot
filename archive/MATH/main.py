import torch
import tensorflow
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import PunktSentenceTokenizer

Tokenizer_wrd = TreebankWordTokenizer()
print(Tokenizer_wrd.tokenize(
  'How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?'
))
word_stemmer = PorterStemmer()
print(word_stemmer.stem('writing'))