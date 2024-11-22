## IndiaAI CyberGuard AI: Cybercrime Prevention Hackathon 

## Team Members 

1. Aditi Singh (Maharaja Agrasen Institute of Technology, Dept. of AI&ML) 
2. Riva Jain (Maharaja Agrasen Institute of Technology, Dept. of AI&ML)
3. Palak Garg (Maharaja Agrasen Institute of Technology, Dept. of AI&ML)

## Introduction
The objective of this project is to harness the power of the BERT (Bidirectional Encoder Representations from Transformers) model, an advanced open-source machine learning framework for Natural Language Processing (NLP), to achieve state-of-the-art performance in multiclass text classification. Specifically, this project focuses on classifying cyber security complaints into relevant categories and subcategories of crime. By automating the process of accurately categorizing these complaints, we aim to enhance the efficiency and effectiveness of fraud detection systems.

To achieve this, we undertook several key steps in the development of our model:
1. **Exploratory Data Analysis (EDA)**: To identify patterns and extract valuable insights from the textual data.
2. **Text Preprocessing**: Cleaning, tokenization, and normalization of the data to prepare it for model training.
3. **Model Development**: Fine-tuning pre-trained BERT models to classify text descriptions into their respective categories and subcategories, ensuring high precision and reliability in the fraud detection system.

## Exploratory Data Analysis (EDA)

In this NLP project, we performed Exploratory Data Analysis (EDA) to better understand the dataset and prepare it for modeling. This step involved identifying patterns, trends, and anomalies in the data, as well as ensuring data quality. EDA provided crucial insights into the distribution of categories, sub-categories, and data imbalances, enabling informed preprocessing decisions.

### Overview of the Dataset
The dataset, loaded from a CSV file named `train.csv`, contains information about various cyber crimes reported in India. The primary columns include:
- `category`: The broad classification of the crime.
- `sub_category`: More specific types of crimes within each category.
- `crime_additional_info`: Additional details or descriptions related to each crime report.

### Key Findings from the EDA
- **Data Inspection**: 
  - The dataset consists of 93,686 records.
  - There are 15 unique categories of crimes and 35 unique sub-categories.
  - The most frequently reported category is Online Financial Fraud, with 57,434 occurrences, followed by various sub-categories such as UPI Related Frauds (26,856 occurrences).

- **Insights into Specific Crimes**: 
  - Fraud CallVishing and Internet Banking Related Fraud are notable sub-categories with high frequencies.
  - Other significant entries include various forms of cyber bullying, online gambling, and job fraud.

### Visualizations:
- Figure 1: Overview of DataFrame Columns
- Figure 2: Summary Statistics: Unique Values and Most Frequent Categories, Sub-Categories, and Additional Information in Crime Data

## Text Preprocessing

We followed a comprehensive text preprocessing pipeline to ensure the quality and consistency of the data before feeding it into the model. These preprocessing steps were crucial in enhancing the model’s ability to understand and classify the text accurately.

### Preprocessing Steps:
1. **Data Imputation**: Handled missing or incomplete data by imputing values to maintain data integrity.
2. **Lowercasing**: Converted all text data to lowercase to eliminate inconsistencies caused by case sensitivity.
3. **Stop Words Removal**: Removed common stop words (e.g., "the," "and," "is") that do not contribute significant meaning to the classification task.
4. **Punctuation Removal**: Stripped punctuation marks from the text as they do not carry essential information for the classification model.
5. **Lemmatization**: Reduced words to their base form (e.g., "running" to "run") to improve model efficiency.
6. **Tokenization**: Split the text into individual words or tokens for further analysis.
7. **POS Tagging**: Identified the grammatical structure of sentences and the role of each word (e.g., noun, verb).
8. **Named Entity Recognition (NER)**: Detected and classified proper nouns (names of individuals, organizations, locations) to provide important context for categorizing complaints.

By performing these preprocessing steps, we ensured that the data was well-prepared for the model, making it easier for BERT to process and classify the text accurately.

## Model Development

The goal of the project is to leverage BERT for state-of-the-art multiclass text classification over cyber security complaints.

### BERT Model Overview
BERT is a transformer-based model that uses an encoder-only architecture. It processes text bidirectionally, which means it considers the context on both sides of a word simultaneously rather than sequentially from left to right or right to left. This approach enhances the model's ability to understand the context of words in a sentence.

### BERT Pre-Training
BERT is pre-trained on large amounts of unlabeled text data to learn contextual embeddings, which represent words in the context of their surrounding words. During pre-training, BERT learns tasks like:
1. **Masked Language Modeling (MLM)**: Predicting missing words in a sentence by considering the surrounding context.
2. **Next Sentence Prediction (NSP)**: Determining whether two sentences are related.

### BERT Fine-Tuning
After pre-training, BERT is fine-tuned on labeled data specific to the downstream task. In our case, we fine-tuned the model to classify cyber security complaints into appropriate categories and sub-categories.

### Workflow
- **Input**: Sequence of tokens representing the complaint text.
- **Processing**: Tokens are passed through the BERT model, which generates contextualized representations for each token.
- **Output**: A sequence of vectors representing the classification prediction for each input token.

### Training Parameters:
- `num_train_epochs`: Number of times the entire dataset is passed through the model during training.
- `per_device_train_batch_size`: Batch size for training.
- `learning_rate`: Initial learning rate for the optimizer.
- `warmup_steps`: Number of steps during which the learning rate linearly increases.
- `logging_steps`: Frequency of logging training metrics.
- `save_steps`: Frequency of saving the model checkpoints.

## Tech Stack

- **Language**: Python
- **Libraries**: 
  - pandas
  - torch
  - nltk
  - numpy
  - pickle
  - re
  - tqdm
  - sklearn
  - transformers

## Prerequisites

To run this project, you will need:
- Python 3.x
- PyTorch
- Transformers library from Hugging Face

Install required libraries and clone the repo to initiate file for the model using tensorflow through the following commands:
```bash
pip install -r requirements.txt
git clone https://github.com/aditisinghh17/IndiaAI_CyberGuardHackathon-.git
python -m tensorflow.keras.models load_model "bert-unbase-case-model.h5"


