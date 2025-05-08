# Fake News Detection System

An advanced fake news detection system that uses multiple text analysis features to identify potentially fake news articles.

## Features

- **Multiple Analysis Methods**:
  - TF-IDF word pattern analysis
  - BERT contextual understanding
  - Sentiment analysis (TextBlob and VADER)
  - Readability metrics

- **User-Friendly Analysis**:
  - Overall prediction (Real/Fake)
  - Confidence score
  - Writing style analysis
  - Emotional tone breakdown
  - Text complexity assessment
  - Potential red flags

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python detection.py
```

2. Analyze news articles:
```bash
python predict_news.py
```

You can then:
- Enter a news title and text manually
- Provide a URL to analyze an online article

## Model Features

The system analyzes articles using:
- Word patterns and frequency
- Contextual meaning
- Emotional content
- Writing style and complexity
- Readability metrics

## Requirements

See `requirements.txt` for full list of dependencies.

## License

MIT License 