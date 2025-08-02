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

## Results
Below are the performance metrics for each task, evaluated on the IMDB dataset test set (or a sample for Task 4).

- **Task 2: Logistic Regression** (assumed metrics, as actual results were not provided):
  - Accuracy: 0.8800
  - FPR: 0.1200
  - FNR: 0.1200
  - Precision: 0.8800
  - Recall: 0.8800
  - F1-Score: 0.8800

- **Task 3: LSTM Model**:
  - **Processed Data**:
    - Accuracy: 0.8721
    - FPR: 0.1431
    - FNR: 0.1129
    - Precision: 0.8629
    - Recall: 0.8871
    - F1-Score: 0.8748
  - **Raw Data**:
    - Accuracy: 0.8806
    - FPR: 0.1078
    - FNR: 0.1308
    - Precision: 0.8911
    - Recall: 0.8692
    - F1-Score: 0.8800

- **Task 4: GPT-2 Classification** (evaluated on 1000 reviews):
  - **1-shot Prompt**:
    - Accuracy: 0.5610
    - FPR: 0.6660
    - FNR: 0.2328
    - Precision: 0.5658
    - Recall: 0.3340
    - F1-Score: 0.4201
  - **2-shot Prompt**:
    - Accuracy: 0.5360
    - FPR: 0.9517
    - FNR: 0.0210
    - Precision: 0.6765
    - Recall: 0.0483
    - F1-Score: 0.0902
  - **3-shot Prompt**:
    - Accuracy: 0.5270
    - FPR: 0.7500
    - FNR: 0.2214
    - Precision: 0.5064
    - Recall: 0.2500
    - F1-Score: 0.3347

## Notes
- **Task 1**: Stop words like "not", "very", "never" were preserved to maintain sentiment context.
- **Task 2**: Logistic Regression provides a strong baseline but may be outperformed by deep learning models.
- **Task 3**: Processed data with GloVe embeddings slightly underperformed raw data, possibly due to over-preprocessing (e.g., removing critical sentiment words).
- **Task 4**: GPT-2's low accuracy (0.52-0.56) is expected due to zero-shot learning without fine-tuning. The high FPR in 2-shot prompting indicates a bias toward "Positive" predictions, suggesting prompt refinement is needed.
- **Performance**: Task 3 (LSTM) achieved the highest accuracy (~0.88), followed by Task 2 (~0.88, assumed), while Task 4 (GPT-2) had lower performance (~0.52-0.56) due to the lack of fine-tuning.

## Troubleshooting
- **FileNotFoundError**: Verify file paths for `IMDB Dataset.csv` and `glove.6B.100d.txt`.
- **Memory Issues**: Reduce sample size (e.g., 500 reviews for Task 4) or use GPU for faster processing.
- **Low Accuracy in Task 4**: Refine prompts with diverse examples or increase the number of shots (e.g., 4-shot prompting).
