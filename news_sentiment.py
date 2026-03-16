from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

def sentiment_score(headlines):

    scores=[]

    for h in headlines:

        result = sentiment(h)[0]

        scores.append(result["score"])

    return sum(scores)/len(scores)
