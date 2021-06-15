import pandas as pd 
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#READ AND WRANGLE DATA
df_data = pd.read_csv('avatar.csv', engine='python')
print(df_data.head())
#print(df_data.describe())
df_data_lines = df_data.groupby('character').count()
df_data_lines = df_data_lines.sort_values(by=['character_words'], ascending=False)[:10]
top_character_names = df_data_lines.index.values
#print(top_character_names)

print("================")
#FILTER OUT NON TOP CHARACTERS 
df_character_sentiment = df_data[df_data['character'].isin(top_character_names)]
df_character_sentiment = df_character_sentiment[['character', 'character_words']]
#print(df_character_sentiment)

#SENTIMENT SCORE
s = SentimentIntensityAnalyzer()
df_character_sentiment.reset_index(inplace=True, drop=True)
df_character_sentiment[['neg', 'neu', 'pos', 'compound']] = df_character_sentiment['character_words'].apply(s.polarity_scores).apply(pd.Series)
print(df_character_sentiment)