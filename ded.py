import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the CSV file with tweets
csv_file_path = 'tweets.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Load the sentiment analysis model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Define labels
labels = ['Negative', 'Neutral', 'Positive']

# Create a list to store the sentiment results
sentiment_results = []

# Iterate through each row of the CSV file
for index, row in df.iterrows():
    tweet = row['tweet']  # Assuming the column name is 'tweet' in your CSV
    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment_result = {}
    for i in range(len(scores)):
        sentiment_label = labels[i]
        sentiment_score = scores[i]
        sentiment_result[sentiment_label] = sentiment_score.item()

    sentiment_results.append(sentiment_result)

# Create a DataFrame to store the sentiment results
sentiment_df = pd.DataFrame(sentiment_results)

# Add the sentiment results to the original DataFrame
df = pd.concat([df, sentiment_df], axis=1)

# Save the DataFrame with sentiment results to a new CSV file
output_csv_file_path = 'tweets_with_sentiment.csv'  # Specify the output file path
df.to_csv(output_csv_file_path, index=False)

print("Sentiment analysis results saved to:", output_csv_file_path)
