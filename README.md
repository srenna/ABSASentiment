# Aspect Based Sentiment Analysis (ABSA)
 The developed function will take as input a list of reviews and return a dataframe with the review id (i.e. the index of the review in the list), dish, corresponding sentiment score, and confidence.

To use, import the function from the submitted PY file. 

```
from absa import absa
reviews = ["the first restaurant review. it can contain multiple sentences.",
           "the second restaurant review",
           ...]
results = absa(reviews)
```
The script will return a dataframe with the results!
