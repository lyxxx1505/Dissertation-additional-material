# Dissertation-additional-material
Repository Structure
This repository contains supplementary materials for the study of ESG news sentiment analysis and topic modeling. It is organized into three main folders:
(1) 'Datasets' folder, Contains 19 raw datasets of ESG-related news articles retrieved from The Guardian API. The keywords used include “ESG”, “environmental, social and governance”, “green finance”, and others. Each dataset is named according to the corresponding keyword(s). 
For example:To search for “responsible investment”, the code parameter was set as: QUERY = '"responsible investment"'
The output was saved as: csv_path = "guardian_responsibleinvestment_2018_2025_analytics.csv"
In total, 19 keywords were used to generate datasets. Note that “esg”, “environmental, social and governance” and “environmental social and governance” are combined into a single dataset.
(2) 'Codes' folder, Contains the core Python scripts used for data collection, data merge, pre-processing, topic modeling, and sentiment analysis:
dataset_code.py: Retrieves ESG news data from The Guardian API by looping over keywords. Outputs datasets named after the keywords.
dataset_merge.py: Merges the 19 datasets into a single file, removes duplicates, and filters out irrelevant articles not related to ESG. The cleaned dataset is then used for LDA topic modeling and sentiment analysis.
lda_code.py: Performs LDA topic modeling. Includes:Text preprocessing, Building the LDA model, Selecting the optimal number of topics,Extracting the top-10 keywords for each topic, Visualizing ESG news volume and topic dynamics over time, Outputting subsets of data by topic (Topic 2 vs. non-Topic 2), Code is annotated for clarity.

sentiment_analysis.py: Conducts sentiment analysis (using VADER) and keyword extraction (using TF-IDF). Includes: Data preprocessing, Overall sentiment analysis and keyword extraction, Sentiment and keyword extraction for Topic 2 and non-Topic 2 subsets, Code contains explanations for key steps.

(3) 'Output' folder, Contains example results generated from running the scripts, including:
Visualizations (e.g., word clouds, sentiment distribution charts, topic evolution line charts)
Example processed datasets

Notes: 
All scripts are written in Python and include comments for reproducibility.
Ensure that dependencies (gensim, nltk, matplotlib, wordcloud, sklearn, vaderSentiment, etc.) are installed before running.

requirements：
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
gensim
scikit-learn
vaderSentiment
pyLDAvis
