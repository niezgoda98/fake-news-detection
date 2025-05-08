import joblib
import requests
from bs4 import BeautifulSoup
from detection import preprocess, TextFeatureExtractor
import numpy as np

def load_model():
    """Load the trained model and feature extractor"""
    try:
        model = joblib.load('fake_news_model.joblib')
        feature_extractor = joblib.load('feature_extractor.joblib')
        return model, feature_extractor
    except FileNotFoundError:
        print("Error: Model files not found. Please run detection.py first to train the model.")
        return None, None

def analyze_features(text, feature_extractor):
    """Analyze and return detailed feature information"""
    # Get BERT embeddings
    bert_features = feature_extractor.get_bert_embeddings(text)
    
    # Get sentiment features
    sentiment_features = feature_extractor.get_sentiment_features(text)
    
    # Get readability features
    readability_features = feature_extractor.get_readability_features(text)
    
    return {
        'sentiment': {
            'textblob': {
                'polarity': sentiment_features['textblob_polarity'],
                'subjectivity': sentiment_features['textblob_subjectivity']
            },
            'vader': {
                'compound': sentiment_features['vader_compound'],
                'positive': sentiment_features['vader_pos'],
                'negative': sentiment_features['vader_neg'],
                'neutral': sentiment_features['vader_neu']
            }
        },
        'readability': {
            'flesch_kincaid_grade': readability_features['flesch_kincaid_grade'],
            'flesch_reading_ease': readability_features['flesch_reading_ease'],
            'smog_index': readability_features['smog_index'],
            'coleman_liau_index': readability_features['coleman_liau_index'],
            'automated_readability_index': readability_features['automated_readability_index'],
            'dale_chall_readability_score': readability_features['dale_chall_readability_score'],
            'difficult_words': readability_features['difficult_words'],
            'linsear_write_formula': readability_features['linsear_write_formula'],
            'gunning_fog': readability_features['gunning_fog']
        }
    }

def predict_from_text(title, text):
    """Predict if news is fake or real from title and text"""
    model, feature_extractor = load_model()
    if model is None or feature_extractor is None:
        return
    
    # Combine title and text
    combined_text = f"{title} {text}"
    processed_text = preprocess(combined_text)
    
    # Get detailed feature analysis
    feature_analysis = analyze_features(processed_text, feature_extractor)
    
    # Extract features for prediction
    features = feature_extractor.extract_features([processed_text])
    
    # Scale features
    features_scaled = feature_extractor.scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': 'Real' if prediction == 1 else 'Fake',
        'confidence': probability[prediction] * 100,
        'analysis': feature_analysis
    }

def predict_from_url(url):
    """Predict if news is fake or real from URL"""
    try:
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find title and article text
        title = soup.find('title').text if soup.find('title') else ""
        
        # Try to find the main article content
        article_text = ""
        for p in soup.find_all('p'):
            article_text += p.text + " "
        
        if not article_text:
            print("Warning: Could not extract article text from the URL")
            return None
            
        return predict_from_text(title, article_text)
        
    except Exception as e:
        print(f"Error processing URL: {str(e)}")
        return None

def print_analysis(result):
    """Print detailed analysis of the prediction in a user-friendly way"""
    print("\n=== Prediction Result ===")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    
    print("\n=== Writing Style Analysis ===")
    # Sentiment Analysis
    sentiment = result['analysis']['sentiment']
    print("\nEmotional Tone:")
    # TextBlob Polarity
    polarity = sentiment['textblob']['polarity']
    if polarity > 0.3:
        tone = "Positive"
    elif polarity < -0.3:
        tone = "Negative"
    else:
        tone = "Neutral"
    print(f"- Overall Tone: {tone}")
    
    # Subjectivity
    subjectivity = sentiment['textblob']['subjectivity']
    if subjectivity > 0.6:
        style = "Very Opinionated"
    elif subjectivity > 0.4:
        style = "Somewhat Opinionated"
    else:
        style = "Factual"
    print(f"- Writing Style: {style}")
    
    # VADER Analysis
    vader = sentiment['vader']
    print("\nEmotional Breakdown:")
    print(f"- Positive Content: {vader['positive']*100:.1f}%")
    print(f"- Negative Content: {vader['negative']*100:.1f}%")
    print(f"- Neutral Content: {vader['neutral']*100:.1f}%")
    
    # Readability Analysis
    readability = result['analysis']['readability']
    print("\nText Complexity:")
    
    # Flesch Reading Ease
    ease = readability['flesch_reading_ease']
    if ease >= 90:
        level = "Very Easy (5th grade)"
    elif ease >= 80:
        level = "Easy (6th grade)"
    elif ease >= 70:
        level = "Fairly Easy (7th grade)"
    elif ease >= 60:
        level = "Standard (8th-9th grade)"
    elif ease >= 50:
        level = "Fairly Difficult (10th-12th grade)"
    elif ease >= 30:
        level = "Difficult (College)"
    else:
        level = "Very Difficult (College Graduate)"
    print(f"- Reading Level: {level}")
    
    # Difficult Words
    diff_words = readability['difficult_words']
    if diff_words < 10:
        complexity = "Simple"
    elif diff_words < 30:
        complexity = "Moderate"
    else:
        complexity = "Complex"
    print(f"- Vocabulary Complexity: {complexity} ({diff_words} difficult words)")
    
    # Gunning Fog Index
    fog = readability['gunning_fog']
    if fog < 6:
        fog_level = "Very Easy"
    elif fog < 12:
        fog_level = "Easy"
    elif fog < 17:
        fog_level = "Moderate"
    else:
        fog_level = "Difficult"
    print(f"- Overall Complexity: {fog_level}")
    
    print("\n=== Key Indicators ===")
    # Print potential red flags
    red_flags = []
    
    # Check for extreme subjectivity
    if subjectivity > 0.7:
        red_flags.append("Highly opinionated content")
    
    # Check for emotional manipulation
    if abs(polarity) > 0.7:
        red_flags.append("Strong emotional language")
    
    # Check for complexity issues
    if fog > 20:
        red_flags.append("Unusually complex language")
    elif fog < 6 and diff_words > 20:
        red_flags.append("Mismatched complexity levels")
    
    if red_flags:
        print("Potential concerns:")
        for flag in red_flags:
            print(f"- {flag}")
    else:
        print("No major concerns detected in the writing style.")

def main():
    print("Fake News Detection System")
    print("-------------------------")
    print("1. Predict from title and text")
    print("2. Predict from URL")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            title = input("Enter the news title: ")
            text = input("Enter the news text: ")
            result = predict_from_text(title, text)
            if result:
                print_analysis(result)
                
        elif choice == "2":
            url = input("Enter the news URL: ")
            result = predict_from_url(url)
            if result:
                print_analysis(result)
                
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()