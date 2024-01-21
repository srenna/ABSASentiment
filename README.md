# ABSA_sentiment
 The function provided will take as input a list of sentences and return a dataframe with the review id (i.e. the index of the review in the list), dish, corresponding sentiment score and confidence.

Import the function from the submitted PY file. 

from absa import absa
reviews = ["the first restaurant review. it can contain multiple sentences.",
           "the second restaurant review",
           ...]
results = absa(reviews)

The script will return a dataframe with the results!
