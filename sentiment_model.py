# ============================================================
# SENTIMENT ANALYSIS USING SPACY + TF-IDF + LOGISTIC REGRESSION
# ============================================================

# -----------------------
# 1. IMPORT LIBRARIES
# -----------------------

import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------
# 2. LOAD SPACY MODEL
# -----------------------

# This loads spaCy's small English language model
nlp = spacy.load("en_core_web_sm")


# -----------------------
# 3. KEEP IMPORTANT NEGATION WORDS
# -----------------------

# Words like "not" are very important for sentiment.
# Example:
# "This product is good"
# "This product is not good"
#
# Therefore, we don't want spaCy to remove these words
# as stop words.

nlp.vocab["not"].is_stop = False
nlp.vocab["no"].is_stop = False
nlp.vocab["never"].is_stop = False


# -----------------------
# 4. CREATE SAMPLE DATASET
# -----------------------

# Later, I can replace this with a CSV file
# containing real Amazon customer reviews.

data = {

    "review": [

        # POSITIVE REVIEWS

        "This router is amazing",
        "Excellent internet speed",
        "The signal range is very good",
        "Very easy to install",
        "I love this product",
        "The router works perfectly",
        "Amazing WiFi performance",
        "The connection is very stable",
        "The internet speed is excellent",
        "Very good product",
        "I am happy with this router",
        "The setup process was very easy",
        "Excellent signal strength",
        "The router has great coverage",
        "I highly recommend this product",

        # NEGATIVE REVIEWS

        "Worst router ever",
        "Internet keeps disconnecting",
        "Very poor signal range",
        "The router is terrible",
        "I hate this product",
        "Very difficult to install",
        "The WiFi connection is very bad",
        "The router keeps losing connection",
        "Internet speed is extremely slow",
        "Very bad product",
        "I am disappointed with this router",
        "The setup process is confusing",
        "Poor signal strength",
        "The router has terrible coverage",
        "I do not recommend this product"
    ],

    "sentiment": [

        # POSITIVE LABELS

        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",

        # NEGATIVE LABELS

        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]
}


# -----------------------
# 5. CONVERT DATA TO DATAFRAME
# -----------------------

# A DataFrame is like a table or Excel sheet in Python

df = pd.DataFrame(data)
# I commented out the following lines because they were printing the original dataset, which is not necessary for the final code. You can uncomment them if you want to see the original dataset.
# print("\n================================")
# print("ORIGINAL DATASET")
# print("================================")

# print(df)


# -----------------------
# 6. CREATE TEXT PREPROCESSING FUNCTION
# -----------------------

def preprocess_text(text):

    # Convert text to lowercase
    # Then process it using spaCy

    doc = nlp(text.lower())

    # Create an empty list
    # We will store useful words here

    tokens = []

    # Go through every token (word)

    for token in doc:

        # Ignore:
        # 1. Stop words
        # 2. Punctuation
        # 3. Spaces

        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
        ):

            # Get the lemma (base form) of the word
            # Example:
            # working -> work
            # routers -> router

            tokens.append(token.lemma_)

    # Convert list back into a sentence

    return " ".join(tokens)


# -----------------------
# 7. CLEAN ALL REVIEWS
# -----------------------

df["clean_review"] = df["review"].apply(preprocess_text)


# print("\n================================")
# print("CLEANED DATASET")
# print("================================")

# print(df[["review", "clean_review"]])


# -----------------------
# 8. CREATE INPUT (X) AND OUTPUT (y)
# -----------------------

# X = Input data (customer reviews)
# y = Output labels (positive or negative)

X = df["clean_review"]

y = df["sentiment"]


# -----------------------
# 9. SPLIT DATA
# -----------------------

# 80% data -> Training
# 20% data -> Testing

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    # Keeps similar proportion of positive
    # and negative examples
    stratify=y
)


# -----------------------
# 10. CREATE TF-IDF VECTORIZER
# -----------------------

# Machine learning models cannot directly understand words.
#
# TF-IDF converts text into numerical features.

vectorizer = TfidfVectorizer()


# -----------------------
# 11. TRAIN TF-IDF
# -----------------------

# fit_transform()
#
# fit -> Learn vocabulary from training data
# transform -> Convert text into numbers

X_train_vectorized = vectorizer.fit_transform(X_train)


# For testing data we ONLY use transform()
#
# We don't want TF-IDF to learn anything
# from the testing data.

X_test_vectorized = vectorizer.transform(X_test)


# -----------------------
# 12. CREATE MACHINE LEARNING MODEL
# -----------------------

model = LogisticRegression()


# -----------------------
# 13. TRAIN MODEL
# -----------------------

# The model learns patterns between:
#
# Review text
#      ↓
# Positive / Negative

model.fit(X_train_vectorized, y_train)


# -----------------------
# 14. TEST MODEL
# -----------------------

predictions = model.predict(X_test_vectorized)


# -----------------------
# 15. CALCULATE ACCURACY
# -----------------------

accuracy = accuracy_score(y_test, predictions)

# print("\n================================")
# print("MODEL PERFORMANCE")
# print("================================")

# print("Accuracy:", accuracy)

# print("\nDetailed Report:")

# print(
#     classification_report(
#         y_test,
#         predictions,
#         zero_division=0
#     )
# )


# -----------------------
# 16. CREATE PREDICTION FUNCTION
# -----------------------

def predict_sentiment(review):

    # Step 1:
    # Clean the new review using spaCy

    cleaned_review = preprocess_text(review)


    # Step 2:
    # Convert cleaned review into numbers
    # using the SAME TF-IDF vectorizer

    vectorized_review = vectorizer.transform(
        [cleaned_review]
    )


    # Step 3:
    # Ask the trained model to predict

    prediction = model.predict(
        vectorized_review
    )[0]


    # Step 4:
    # Get probability/confidence

    probabilities = model.predict_proba(
        vectorized_review
    )[0]


    # Find highest probability

    confidence = max(probabilities)


    # Return prediction and confidence

    return prediction, confidence


# -----------------------
# 17. TEST WITH SAMPLE REVIEWS
# -----------------------

# print("\n================================")
# print("SAMPLE PREDICTIONS")
# print("================================")


test_reviews = [

    "This router is excellent",

    "The internet connection is terrible",

    "I love the WiFi speed",

    "The signal range is very poor",

    "This is the worst router I have ever used"

]


for review in test_reviews:

    sentiment, confidence = predict_sentiment(review)

    # print("\nReview:", review)

    # print("Sentiment:", sentiment)

    # print(
    #     "Confidence:",
    #     round(confidence * 100, 2),
    #     "%"
    # )


# -----------------------
# 18. ALLOW USER TO ENTER REVIEWS
# -----------------------

print("\n================================")
print("SENTIMENT ANALYSIS SYSTEM")
print("================================")

print("Type 'exit' to stop the program.")


while True:

    # Ask user for a review

    user_review = input(
        "\nEnter a customer review: "
    )


    # Stop program if user types exit

    if user_review.lower() == "exit":

        print(
            "\nSentiment Analysis System Closed."
        )

        break


    # Predict sentiment

    sentiment, confidence = predict_sentiment(
        user_review
    )


    # Display result

    print("\n----------------------------")

    print(
        "Sentiment:",
        sentiment.upper()
    )

    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )

    print("----------------------------")
