# Import necessary libraries
import spacy  # Import spaCy library for natural language processing
# Import the Aspect-Based Sentiment Analysis (ABSA) checkpoint manager
from pyabsa import APCCheckpointManager
import pandas as pd  # Import pandas library for data manipulation

# Define the absa function that performs Aspect-Based Sentiment Analysis (ABSA)


def absa(reviews):
    # Obtain the sentiment classifier model from the APC checkpoint manager
    sentiment_classifier = APCCheckpointManager.get_sentiment_classifier(
        checkpoint='multilingual', auto_device=False)

    # Load the spaCy model for natural language processing
    nlp = spacy.load('model-best')

    # Initialize an empty list to store the ABSA results
    results_list = []

    # Loop through each review in the provided list
    for review_id, review in enumerate(reviews):
        # Process the review text using the spaCy NLP pipeline
        doc = nlp(review)

        # Extract dish entities from the review using spaCy's named entity
        # recognition (NER)
        dishes = [ent.text for ent in doc.ents if ent.label_ == "DISH"]

        # Loop through each identified dish in the review
        for dish in dishes:
            # Tag the dish in the review text with aspect markers [B-ASP] and
            # [E-ASP]
            tagged_review = review.replace(dish, f"[B-ASP]{dish}[E-ASP]")

            try:
                # Perform sentiment inference on the tagged review using the
                # sentiment classifier
                result = sentiment_classifier.infer(text=tagged_review,
                                                    print_result=False,
                                                    ignore_error=True,
                                                    clear_input_samples=True)

                # Check if the required indices (aspect, sentiment, confidence)
                # exist in the result
                if 'aspect' in result and 'sentiment' in result and 'confidence' in result:
                    # Append the ABSA results to the results_list in a
                    # dictionary format
                    results_list.append({
                        'review_id': review_id,
                        'dish': result['aspect'][0],
                        'sentiment': result['sentiment'][0],
                        'confidence': round(result['confidence'][0], 2),
                    })

                else:
                    # Print a message if any of the required indices are
                    # missing in the result
                    print(
                        f"Skipping result for review_id {review_id} due to missing indices.")

            # Handle exceptions or errors that might occur during sentiment
            # inference
            except Exception as e:
                print(f"Error processing review_id {review_id}: {str(e)}")

    # Create a pandas DataFrame from the results_list and return it as the
    # output
    result_df = pd.DataFrame(results_list)
    return result_df  # Return the DataFrame containing ABSA results
