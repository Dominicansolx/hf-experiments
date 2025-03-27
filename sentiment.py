from transformers import pipeline

# Set up sentiment analysis
classifier = pipeline("sentiment-analysis")

# Test it
text = "I love Hugging Face!"
result = classifier(text)

# Show results
print(f"Text: {text}")
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")