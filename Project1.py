import pandas as pd
import torch
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("./emotion_detection_roberta_model")
tokenizer = RobertaTokenizer.from_pretrained("./emotion_detection_roberta_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load Amazon data
amazon_df = pd.read_csv('amazon_reviews.csv')

# Define emotion labels
label_names = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Define emotion-to-sentiment mappings
EMOTION_TO_SENTIMENT = {
    "admiration": "Positive", "amusement": "Positive", "approval": "Positive",
    "caring": "Positive", "excitement": "Positive", "gratitude": "Positive",
    "joy": "Positive", "love": "Positive", "optimism": "Positive",
    "pride": "Positive", "relief": "Positive", "desire": "Positive",
    "anger": "Negative", "annoyance": "Negative", "disappointment": "Negative",
    "disapproval": "Negative", "disgust": "Negative", "embarrassment": "Negative",
    "fear": "Negative", "grief": "Negative", "nervousness": "Negative",
    "remorse": "Negative", "sadness": "Negative",
    "confusion": "Neutral", "curiosity": "Neutral", "realization": "Neutral",
    "surprise": "Neutral", "neutral": "Neutral"
}

EMOTION_TO_FINE_SENTIMENT = {
    "admiration": "Positive", "amusement": "Positive", "approval": "Positive",
    "caring": "Positive", "excitement": "Very Positive", "gratitude": "Very Positive",
    "joy": "Very Positive", "love": "Very Positive", "optimism": "Positive",
    "pride": "Positive", "relief": "Positive", "desire": "Positive",
    "anger": "Very Negative", "annoyance": "Negative", "disappointment": "Negative",
    "disapproval": "Negative", "disgust": "Very Negative", "embarrassment": "Negative",
    "fear": "Very Negative", "grief": "Very Negative", "nervousness": "Negative",
    "remorse": "Negative", "sadness": "Negative",
    "confusion": "Neutral", "curiosity": "Neutral", "realization": "Neutral",
    "surprise": "Neutral", "neutral": "Neutral"
}


def predict_emotions_with_confidence(texts, model, tokenizer, device, batch_size=16):
    emotions_list = []
    confidence_list = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_scores = torch.sigmoid(logits).cpu().numpy()
            batch_preds = (batch_scores > 0.5).astype(int)

        for j in range(len(batch_texts)):
            emotion_confidences = []
            active_emotions = []

            for idx, val in enumerate(batch_preds[j]):
                if val:
                    emotion = label_names[idx]
                    confidence = batch_scores[j][idx]
                    emotion_confidences.append((emotion, confidence))
                    active_emotions.append(emotion)

            emotion_confidences.sort(key=lambda x: x[1], reverse=True)
            emotions_list.append(active_emotions)
            confidence_list.append(emotion_confidences)

    return emotions_list, confidence_list


def calculate_sentiment(emotions, confidences):
    if not emotions:
        return {
            "sentiment_polarity": "Neutral",
            "sentiment_fine": "Neutral",
            "sentiment_confidence": 1.0
        }

    polarity_scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
    for emotion, confidence in confidences:
        sentiment = EMOTION_TO_SENTIMENT.get(emotion, "Neutral")
        polarity_scores[sentiment] += confidence

    total_score = sum(polarity_scores.values())
    if total_score > 0:
        for sentiment in polarity_scores:
            polarity_scores[sentiment] /= total_score

    dominant_polarity = max(polarity_scores, key=polarity_scores.get)
    polarity_confidence = polarity_scores[dominant_polarity]

    fine_sentiments = {}
    for emotion, confidence in confidences:
        sentiment = EMOTION_TO_FINE_SENTIMENT.get(emotion, "Neutral")
        fine_sentiments[sentiment] = fine_sentiments.get(sentiment, 0.0) + confidence

    total_fine_score = sum(fine_sentiments.values())
    if total_fine_score > 0:
        for sentiment in fine_sentiments:
            fine_sentiments[sentiment] /= total_fine_score

    dominant_fine = max(fine_sentiments, key=fine_sentiments.get)
    fine_confidence = fine_sentiments[dominant_fine]

    overall_confidence = max(polarity_confidence, fine_confidence)

    return {
        "sentiment_polarity": dominant_polarity,
        "sentiment_fine": dominant_fine,
        "sentiment_confidence": overall_confidence
    }


# Predict emotions and confidences
print("Predicting emotions and confidence scores...")
texts = amazon_df['reviews.text'].tolist()
emotions_list, confidence_list = predict_emotions_with_confidence(
    texts, model, tokenizer, device
)

# Add to DataFrame
amazon_df['emotions'] = emotions_list
amazon_df['emotion_confidences'] = confidence_list

# Calculate sentiment
print("\nCalculating sentiment from emotions...")
sentiment_data = []
for idx, row in tqdm(amazon_df.iterrows(), total=len(amazon_df)):
    sentiment = calculate_sentiment(row['emotions'], row['emotion_confidences'])
    sentiment_data.append(sentiment)

# Add sentiment columns
sentiment_df = pd.DataFrame(sentiment_data)
amazon_df = pd.concat([amazon_df, sentiment_df], axis=1)

# --- Emotion Analysis ---

# Aggregate by brand
brand_emotions = amazon_df.explode('emotions').groupby(['brand', 'emotions']).size().unstack(fill_value=0)

# Top emotions for Amazon brand
if 'Amazon' in brand_emotions.index:
    top_emotions = brand_emotions.loc['Amazon'].sort_values(ascending=False).head(5)

    # Plot top emotions for Amazon brand
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_emotions.values, y=top_emotions.index)
    plt.title('Top Emotions for Amazon')
    plt.xlabel('Count')
    plt.ylabel('Emotion')
    plt.show()

# Temporal emotion analysis
try:
    if 'reviews.date' in amazon_df.columns:
        amazon_df['date'] = pd.to_datetime(
            amazon_df['reviews.date'],
            format='ISO8601',
            errors='coerce'
        )
        amazon_df = amazon_df.dropna(subset=['date'])

        if not amazon_df.empty:
            monthly_emotions = amazon_df.explode('emotions').groupby(
                [pd.Grouper(key='date', freq='ME'), 'emotions']
            ).size().unstack(fill_value=0)

            plt.figure(figsize=(12, 8))
            sns.heatmap(monthly_emotions.T, cmap='Blues')
            plt.title('Monthly Emotion Trends')
            plt.xlabel('Month')
            plt.ylabel('Emotion')
            plt.show()
        else:
            print("No valid dates remaining after cleaning")
    else:
        print("'reviews.date' column not found - skipping temporal analysis")
except Exception as e:
    print(f"Error processing dates: {str(e)}")
    print("Skipping temporal analysis due to date processing issues")

# --- Sentiment Visualization ---

plt.figure(figsize=(14, 6))

# Sentiment polarity distribution
plt.subplot(1, 2, 1)
sns.countplot(data=amazon_df, x='sentiment_polarity',
              order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Fine-grained sentiment distribution
plt.subplot(1, 2, 2)
sns.countplot(data=amazon_df, x='sentiment_fine',
              order=['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'])
plt.title('Fine-grained Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# --- Sentiment Trend Analysis Over Time ---

try:
    if 'date' in amazon_df.columns and not amazon_df['date'].isnull().all():
        # Convert sentiment to numerical values for aggregation
        sentiment_map = {
            'Very Negative': -2,
            'Negative': -1,
            'Neutral': 0,
            'Positive': 1,
            'Very Positive': 2
        }
        amazon_df['sentiment_score'] = amazon_df['sentiment_fine'].map(sentiment_map)

        # Filter out dates that couldn't be parsed
        amazon_df = amazon_df.dropna(subset=['date'])

        if not amazon_df.empty:
            # Create a copy for time series analysis
            time_series_df = amazon_df.copy()
            time_series_df.set_index('date', inplace=True)

            # --- Daily Sentiment Trend ---
            daily_sentiment = time_series_df.resample('D')['sentiment_score'].mean()

            # --- Weekly Sentiment Trend ---
            weekly_sentiment = time_series_df.resample('W')['sentiment_score'].mean()

            # --- Monthly Sentiment Analysis ---
            monthly_sentiment = time_series_df.resample('ME')['sentiment_score'].agg(['mean', 'count'])

            # Create figure
            plt.figure(figsize=(14, 10))

            # Daily sentiment plot
            plt.subplot(3, 1, 1)
            daily_sentiment.plot(color='royalblue', linewidth=1.5)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            plt.title('Daily Brand Sentiment Trend')
            plt.ylabel('Sentiment Score')
            plt.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(np.arange(len(daily_sentiment)), daily_sentiment, 1)
            p = np.poly1d(z)
            plt.plot(daily_sentiment.index, p(np.arange(len(daily_sentiment))),
                     color='red', linestyle='--', label='Trend Line')
            plt.legend()

            # Weekly sentiment plot
            plt.subplot(3, 1, 2)
            weekly_sentiment.plot(color='darkgreen', linewidth=2)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            plt.title('Weekly Brand Sentiment Trend')
            plt.ylabel('Sentiment Score')
            plt.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(np.arange(len(weekly_sentiment)), weekly_sentiment, 1)
            p = np.poly1d(z)
            plt.plot(weekly_sentiment.index, p(np.arange(len(weekly_sentiment))),
                     color='red', linestyle='--', label='Trend Line')
            plt.legend()

            # Monthly sentiment and volume
            plt.subplot(3, 1, 3)
            ax = monthly_sentiment['mean'].plot(kind='bar', color='purple', alpha=0.7, width=0.8)
            plt.title('Monthly Brand Sentiment')
            plt.ylabel('Average Sentiment')
            plt.xlabel('Month')

            # Add review count as line plot
            ax2 = ax.twinx()
            monthly_sentiment['count'].plot(ax=ax2, color='orange', linewidth=2, marker='o')
            ax2.set_ylabel('Review Count', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            # Format x-axis labels
            ax.set_xticklabels([d.strftime('%b %Y') for d in monthly_sentiment.index], rotation=45)

            plt.tight_layout()
            plt.savefig('sentiment_trends_over_time.png')
            plt.show()

            # Print key insights
            print("\n=== Sentiment Trend Insights ===")
            print(f"Overall Average Sentiment: {time_series_df['sentiment_score'].mean():.2f}")
            print(
                f"Highest Monthly Sentiment: {monthly_sentiment['mean'].max():.2f} in {monthly_sentiment['mean'].idxmax().strftime('%B %Y')}")
            print(
                f"Lowest Monthly Sentiment: {monthly_sentiment['mean'].min():.2f} in {monthly_sentiment['mean'].idxmin().strftime('%B %Y')}")

        else:
            print("No valid dates remaining after cleaning - skipping sentiment trend analysis")
    else:
        print("Date information not available - skipping sentiment trend analysis")
except Exception as e:
    print(f"Error in sentiment trend analysis: {str(e)}")
    print("Skipping sentiment trend analysis due to processing issues")

# --- Top Reviews Analysis ---

# Sentiment confidence distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=amazon_df, x='sentiment_confidence', hue='sentiment_polarity',
             element='step', stat='density', common_norm=False, bins=20)
plt.title('Sentiment Confidence Distribution')
plt.xlabel('Confidence Score')
plt.ylabel('Density')
plt.show()

# Top reviews by sentiment confidence
print("\nTop Positive Reviews:")
top_positive = amazon_df[
    (amazon_df['sentiment_polarity'] == 'Positive') &
    (amazon_df['sentiment_confidence'] > 0.9)
    ].sort_values('sentiment_confidence', ascending=False)
print(top_positive[['reviews.text', 'sentiment_fine', 'sentiment_confidence']].head(3))

print("\nTop Negative Reviews:")
top_negative = amazon_df[
    (amazon_df['sentiment_polarity'] == 'Negative') &
    (amazon_df['sentiment_confidence'] > 0.9)
    ].sort_values('sentiment_confidence', ascending=False)
print(top_negative[['reviews.text', 'sentiment_fine', 'sentiment_confidence']].head(3))



# --- Review Clustering & Topic Modeling ---
print("\nPerforming topic modeling with BERTopic...")

try:
    # Prepare text data
    docs = amazon_df['reviews.text'].tolist()

    # Precompute embeddings
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)

    # Initialize BERTopic
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        verbose=True
    )

    # Fit model
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Add topics to DataFrame
    amazon_df['topic'] = topics
    amazon_df['topic_prob'] = probs.max(axis=1)

    # Get topic information
    topic_info = topic_model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info.head(10))

    # Visualize topics
    fig_topic = topic_model.visualize_topics()
    fig_topic.write_html("topic_visualization.html")

    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html("topic_hierarchy.html")

    # --- Sentiment Distribution Across Topics ---
    # Merge topic names into main DataFrame
    topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
    amazon_df['topic_name'] = amazon_df['topic'].map(topic_name_map)

    # Filter out outliers (topic -1)
    valid_topics_df = amazon_df[amazon_df['topic'] != -1]

    # Top 10 topics by count
    top_topics = valid_topics_df['topic'].value_counts().head(10).index.tolist()
    filtered_df = valid_topics_df[valid_topics_df['topic'].isin(top_topics)]

    # Sentiment distribution across topics
    plt.figure(figsize=(14, 8))
    ax = sns.countplot(
        data=filtered_df,
        x='topic_name',
        hue='sentiment_fine',
        hue_order=['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'],
        order=filtered_df['topic_name'].value_counts().index
    )
    plt.title('Sentiment Distribution Across Topics')
    plt.xlabel('Topic')
    plt.ylabel('Review Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('sentiment_by_topic.png')
    plt.show()

    # Topic sentiment analysis
    topic_sentiment = filtered_df.groupby(['topic_name', 'sentiment_fine']).size().unstack()
    topic_sentiment['Total'] = topic_sentiment.sum(axis=1)
    for sentiment in ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']:
        topic_sentiment[f'{sentiment} %'] = (topic_sentiment[sentiment] / topic_sentiment['Total']) * 100

    print("\nTopic Sentiment Analysis:")
    print(topic_sentiment.sort_values('Total', ascending=False))

    # --- Representative Reviews for Each Topic ---
    print("\nRepresentative Reviews for Top Topics:")
    for topic_id in top_topics:
        topic_reviews = filtered_df[filtered_df['topic'] == topic_id]

        # Get most representative reviews (high topic probability)
        rep_reviews = topic_reviews.sort_values('topic_prob', ascending=False).head(3)

        # Get most positive and negative reviews
        positive_reviews = topic_reviews[topic_reviews['sentiment_polarity'] == 'Positive'] \
            .sort_values('sentiment_confidence', ascending=False).head(1)
        negative_reviews = topic_reviews[topic_reviews['sentiment_polarity'] == 'Negative'] \
            .sort_values('sentiment_confidence', ascending=False).head(1)

        print(f"\nTopic {topic_id}: {topic_name_map[topic_id]}")
        print(f"Representative Reviews:")
        for i, row in rep_reviews.iterrows():
            print(f"- {row['reviews.text'][:200]}... (Sentiment: {row['sentiment_fine']})")

        if not positive_reviews.empty:
            print(f"\nMost Positive Review:")
            print(f"- {positive_reviews.iloc[0]['reviews.text'][:200]}...")

        if not negative_reviews.empty:
            print(f"\nMost Negative Review:")
            print(f"- {negative_reviews.iloc[0]['reviews.text'][:200]}...")

    # --- Topic Trends Over Time ---
    if 'date' in amazon_df.columns:
        try:
            # Filter valid topics and dates
            time_topic_df = filtered_df[['date', 'topic_name']].copy()
            time_topic_df = time_topic_df.dropna(subset=['date'])

            if not time_topic_df.empty:
                # Resample to monthly frequency
                monthly_topics = time_topic_df.set_index('date') \
                    .groupby(['topic_name', pd.Grouper(freq='ME')]) \
                    .size().unstack().fillna(0)

                # Normalize by total reviews per month
                monthly_totals = monthly_topics.sum(axis=0)
                monthly_topics = monthly_topics.div(monthly_totals, axis=1) * 100

                # Plot topic trends
                plt.figure(figsize=(14, 8))
                for topic in monthly_topics.head(5).index:
                    plt.plot(monthly_topics.columns, monthly_topics.loc[topic], label=topic)

                plt.title('Monthly Topic Prevalence (%)')
                plt.xlabel('Month')
                plt.ylabel('Percentage of Reviews')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('topic_trends_over_time.png')
                plt.show()
        except Exception as e:
            print(f"Error in topic time trends: {str(e)}")

except Exception as e:
    print(f"Error in topic modeling: {str(e)}")
    print("Skipping topic modeling due to processing issues")

# Save results to CSV (AFTER topic modeling)
amazon_df.to_csv('amazon_reviews_with_sentiment.csv', index=False)
print("\nFinal results saved to 'amazon_reviews_with_sentiment.csv'")
