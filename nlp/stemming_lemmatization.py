import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

'''
#STEMMING
##PorterStemmer
porter = PorterStemmer()
##LancasterStemmer
lancaster = LancasterStemmer()
print(porter.stem("friendship"))
print(lancaster.stem("friendship"))

print(porter.stem("enjoyable"))
print(lancaster.stem("enjoyable"))

print(porter.stem("sympathetic"))
print(lancaster.stem("sympathetic"))

print(porter.stem("disagreeable"))
print(lancaster.stem("disagreeable"))
'''

from nltk import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
words = ['articles', 'friendship', 'studies', 'phones']
for word in words:
    print(lemmatizer.lemmatize(word))

words_other = ['be', 'is', 'are', 'were', 'was']
for word in words_other:
    print(lemmatizer.lemmatize(word, pos='v'))