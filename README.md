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
git clone https://github.com/niezgoda98/fake-news-detection.git
cd fake-news-detection
```

2. Download the required model files:
   - Download `fake_news_model.joblib` and `feature_extractor.joblib` from [Google Drive](https://drive.google.com/drive/folders/1UgiK1Cn5ocRCD-pwatC1aHNJRo2Pc72E?usp=drive_link)
   - Place these files in the root directory of the project

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python download_nltk.py
```

## Usage

1. Analyze news articles:
```bash
python predict_news.py
```

You can then:
- Enter a news title and text manually
- Provide a URL to analyze an online article

## Model Files

The system requires two model files to function:
- `fake_news_model.joblib`: The trained classification model
- `feature_extractor.joblib`: The feature extraction pipeline

These files are not included in the repository due to their size. You can download them from [Google Drive](https://drive.google.com/drive/folders/1UgiK1Cn5ocRCD-pwatC1aHNJRo2Pc72E?usp=drive_link).

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