import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from transformers import BertTokenizer, BertModel
import torch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from tqdm import tqdm

# Download required NLTK data
nltk.download('stopwords')

class TextFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.vader = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer on all texts"""
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_vectorizer.fit(texts)
        self.is_fitted = True
        
    def get_bert_embeddings(self, text):
        try:
            # Tokenize and prepare input
            inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding as sentence representation
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            return embeddings[0]  # Return first (and only) sentence embedding
        except Exception as e:
            print(f"Error in BERT embedding: {str(e)}")
            return np.zeros(768)  # Return zero vector if there's an error
    
    def get_sentiment_features(self, text):
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER sentiment (without translation)
            vader_scores = self.vader.polarity_scores(text)
            
            return {
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neg': vader_scores['neg'],
                'vader_neu': vader_scores['neu']
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                'textblob_polarity': 0,
                'textblob_subjectivity': 0,
                'vader_compound': 0,
                'vader_pos': 0,
                'vader_neg': 0,
                'vader_neu': 0
            }
    
    def get_readability_features(self, text):
        try:
            return {
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'smog_index': textstat.smog_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                'difficult_words': textstat.difficult_words(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'gunning_fog': textstat.gunning_fog(text)
            }
        except Exception as e:
            print(f"Error in readability analysis: {str(e)}")
            return {
                'flesch_kincaid_grade': 0,
                'flesch_reading_ease': 0,
                'smog_index': 0,
                'coleman_liau_index': 0,
                'automated_readability_index': 0,
                'dale_chall_readability_score': 0,
                'difficult_words': 0,
                'linsear_write_formula': 0,
                'gunning_fog': 0
            }
    
    def extract_features(self, texts):
        print("Extracting features...")
        features = []
        
        # Fit TF-IDF if not already fitted
        if not self.is_fitted:
            self.fit_tfidf(texts)
        
        # Transform all texts with TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Define the expected feature dimensions
        tfidf_dim = tfidf_features.shape[1]  # Should be 5000
        bert_dim = 768
        sentiment_dim = 6  # 6 sentiment features
        readability_dim = 9  # 9 readability features
        total_dim = tfidf_dim + bert_dim + sentiment_dim + readability_dim
        
        for i, text in enumerate(tqdm(texts)):
            try:
                # BERT embeddings
                bert_features = self.get_bert_embeddings(text)
                
                # Sentiment features
                sentiment_features = self.get_sentiment_features(text)
                sentiment_array = np.array([
                    sentiment_features['textblob_polarity'],
                    sentiment_features['textblob_subjectivity'],
                    sentiment_features['vader_compound'],
                    sentiment_features['vader_pos'],
                    sentiment_features['vader_neg'],
                    sentiment_features['vader_neu']
                ])
                
                # Readability features
                readability_features = self.get_readability_features(text)
                readability_array = np.array([
                    readability_features['flesch_kincaid_grade'],
                    readability_features['flesch_reading_ease'],
                    readability_features['smog_index'],
                    readability_features['coleman_liau_index'],
                    readability_features['automated_readability_index'],
                    readability_features['dale_chall_readability_score'],
                    readability_features['difficult_words'],
                    readability_features['linsear_write_formula'],
                    readability_features['gunning_fog']
                ])
                
                # Combine all features
                combined_features = np.concatenate([
                    tfidf_features[i],
                    bert_features,
                    sentiment_array,
                    readability_array
                ])
                
                # Verify the shape
                assert combined_features.shape[0] == total_dim, f"Feature vector shape mismatch. Expected {total_dim}, got {combined_features.shape[0]}"
                
                features.append(combined_features)
            except Exception as e:
                print(f"Error processing text {i}: {str(e)}")
                # Add zero features if there's an error
                features.append(np.zeros(total_dim))
        
        # Convert to numpy array and verify final shape
        features_array = np.array(features)
        print(f"Final feature matrix shape: {features_array.shape}")
        return features_array

def preprocess(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Split into words
        words = text.split()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""

def load_and_prepare_data(fake_news_path, real_news_path):
    print("Loading data...")
    # Load the datasets
    fake_df = pd.read_csv(fake_news_path)
    real_df = pd.read_csv(real_news_path)
    
    # Add labels
    fake_df['label'] = 0  # 0 for fake news
    real_df['label'] = 1  # 1 for real news
    
    # Combine the datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Preprocess the text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess)
    
    return df

def train_model(df):
    print("Training model...")
    # Initialize feature extractor
    feature_extractor = TextFeatureExtractor()
    
    # Extract features
    X = feature_extractor.extract_features(df['processed_text'].values)
    y = df['label'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train = feature_extractor.scaler.fit_transform(X_train)
    X_test = feature_extractor.scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and feature extractor
    joblib.dump(model, 'fake_news_model.joblib')
    joblib.dump(feature_extractor, 'feature_extractor.joblib')
    
    return model, feature_extractor

def predict_news(text, model, feature_extractor):
    # Preprocess the input text
    processed_text = preprocess(text)
    
    # Extract features
    features = feature_extractor.extract_features([processed_text])[0]
    
    # Scale features
    features = feature_extractor.scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'prediction': 'Real' if prediction == 1 else 'Fake',
        'confidence': probability[prediction] * 100
    }

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual CSV file paths
    fake_news_path = "Fake.csv"
    real_news_path = "True.csv"
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(fake_news_path, real_news_path)
        
        # Train the model
        model, feature_extractor = train_model(df)
        
        # Example prediction
        test_text = "This is a sample news article to test the model."
        result = predict_news(test_text, model, feature_extractor)
        print(f"\nPrediction for test text: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        
    except FileNotFoundError:
        print("Please make sure both CSV files are in the correct location.")
