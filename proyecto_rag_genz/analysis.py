from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

from rag_pipeline import textos

sia = SentimentIntensityAnalyzer()

for t in textos:
    print(t.strip())
    print(sia.polarity_scores(t))
