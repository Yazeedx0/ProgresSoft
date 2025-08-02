# Movie Reviews Sentiment Analysis Project

## Overview
This project implements a sentiment analysis pipeline for the IMDB dataset of 50,000 movie reviews, classifying them as **Positive** or **Negative**. The project consists of four tasks:
1. **Text Processing**: Preprocessing the raw reviews to clean and normalize text data.
2. **Machine Learning Model**: Training a Logistic Regression model on processed data.
3. **LSTM Model**: Training LSTM models on both processed and raw data, enhanced with GloVe embeddings for processed data.
4. **GPT-2 Classification**: Using GPT-2 with 1-shot, 2-shot, and 3-shot prompting for zero-shot sentiment classification.

The project is implemented in Python on Google Colab, leveraging libraries like pandas, NLTK, scikit-learn, TensorFlow, and Hugging Face Transformers.

## Python Version
Python 3.8 or higher

## Dependencies
Install the required libraries using:
```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
```
pandas==1.5.3
nltk==3.8.1
scikit-learn==1.2.2
tensorflow==2.12.0
transformers==4.35.2
torch==2.0.1
```

## NLTK Resources
Download required NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Dataset
- Download the IMDB dataset from: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/talaksahmi/25npath/lmmb-dataset-of-50k-movie-reviews)
- Place the CSV file (`IMDB Dataset.csv`) in `/content/dataset/` in Google Colab, or update the file path in the scripts.
- For Task 3, download GloVe embeddings (`glove.6B.100d.txt`) from: [Stanford GloVe](http://nlp.stanford.edu/data/glove.6B.zip)
  - Unzip and place in `/content/glove/` or update the path in the script.

## Project Structure
The project is divided into four tasks, each with its own script and outputs. All scripts are designed to run on Google Colab with GPU support for faster training.

### Task 1: Text Processing
- **Objective**: Preprocess movie reviews to remove noise and normalize text.
- **Steps**:
  1. Load `IMDB Dataset.csv`.
  2. Apply preprocessing: convert to lowercase, remove HTML tags, URLs, emails, punctuation, stop words, and apply lemmatization.
  3. Save processed reviews to `processed_reviews.pkl`.
- **Script**: `text_processing.py`
- **Outputs**: `processed_reviews.pkl` (processed reviews and sentiments)

### Task 2: Machine Learning Model
- **Objective**: Train a Logistic Regression model on processed data for sentiment classification.
- **Steps**:
  1. Load `processed_reviews.pkl`.
  2. Convert text to TF-IDF features using `TfidfVectorizer`.
  3. Train a Logistic Regression model and evaluate metrics (Accuracy, FPR, FNR, Precision, Recall, F1-Score).
  4. Save the model and vectorizer.
- **Script**: `ml_model.py`
- **Outputs**:
  - `lr_model.pkl` (trained Logistic Regression model)
  - `tfidf_vectorizer.pkl` (TF-IDF vectorizer)
  - Performance metrics printed to console

### Task 3: LSTM Model
- **Objective**: Train LSTM models on both processed and raw data for sentiment classification.
- **Steps**:
  1. Load `processed_reviews.pkl` (processed data) and `IMDB Dataset.csv` (raw data).
  2. Use `Tokenizer` and padding to prepare text data.
  3. For processed data, integrate GloVe embeddings (`glove.6B.100d.txt`).
  4. Train two LSTM models (one for processed data, one for raw data).
  5. Evaluate metrics (Accuracy, FPR, FNR, Precision, Recall, F1-Score).
  6. Save models and tokenizers.
- **Script**: `lstm_sentiment_classification.py`
- **Outputs**:
  - `lstm_processed_model.h5` (LSTM model for processed data)
  - `lstm_raw_model.h5` (LSTM model for raw data)
  - `tokenizer_processed.pkl` (tokenizer for processed data)
  - `tokenizer_raw.pkl` (tokenizer for raw data)
  - Performance metrics printed to console

### Task 4: GPT-2 Classification
- **Objective**: Use GPT-2 for zero-shot sentiment classification with 1-shot, 2-shot, and 3-shot prompting.
- **Steps**:
  1. Load `IMDB Dataset.csv` (raw data).
  2. Implement 1-shot, 2-shot, and 3-shot prompts with examples.
  3. Classify reviews using GPT-2, ensuring output is strictly "Positive" or "Negative" via robust parsing.
  4. Evaluate on a sample of 1000 reviews and compute metrics (Accuracy, FPR, FNR, Precision, Recall, F1-Score).
- **Script**: `gpt2_classification.py`
- **Outputs**:
  - Performance metrics for each prompting type printed to console
  - `gpt2_classification.py` (classification script)

## Running the Project
1. **Setup**:
   - Ensure the dataset (`IMDB Dataset.csv`) and GloVe embeddings (`glove.6B.100d.txt`) are in the correct paths.
   - Install dependencies using `requirements.txt`.
   - Download NLTK resources.
2. **Execution**:
   - Run `text_processing.py` to generate `processed_reviews.pkl`.
   - Run `ml_model.py` to train and evaluate the Logistic Regression model.
   - Run `lstm_sentiment_classification.py` to train and evaluate the LSTM models.
   - Run `gpt2_classification.py` to classify reviews using GPT-2 and evaluate prompting performance.
3. **GPU Usage**:
   - For faster training (especially for Task 3 and 4), enable GPU in Colab: **Runtime > Change runtime type > GPU**.
4. **File Management**:
   - Outputs are saved in `/content/` and can be copied to Google Drive:
     ```bash
     !cp /content/*.pkl /content/drive/MyDrive/
     !cp /content/*.h5 /content/drive/MyDrive/
     !cp /content/*.py /content/drive/MyDrive/
     ```

## Notes
- **Task 1**: Ensure stop words like "not", "very", "never" are preserved during preprocessing to maintain sentiment context.
- **Task 2**: Logistic Regression is efficient but may underperform compared to deep learning models.
- **Task 3**: Processed data with GloVe embeddings typically outperforms raw data, but results may vary based on preprocessing quality.
- **Task 4**: GPT-2 without fine-tuning may yield lower accuracy (e.g., 0.60-0.75). Improve prompts with diverse examples for better performance.
- **Performance**: Expected metrics (based on IMDB dataset):
  - Task 2 (Logistic Regression): Accuracy ~0.85-0.90
  - Task 3 (LSTM): Accuracy ~0.87-0.89 (processed data likely better)
  - Task 4 (GPT-2): Accuracy ~0.60-0.75 (3-shot better than 1-shot)

## Troubleshooting
- **FileNotFoundError**: Verify file paths for `IMDB Dataset.csv` and `glove.6B.100d.txt`.
- **Memory Issues**: Reduce sample size (e.g., 500 reviews for Task 4) or use GPU for faster processing.
- **Low Accuracy**: For Task 3, try increasing `max_words`, `max_len`, or epochs. For Task 4, refine prompts with better examples.
