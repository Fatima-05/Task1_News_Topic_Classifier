# DevelopersHub AI/ML Engineering Internship Tasks

## Task1: News Topic Classification Using BERT

### Task Objective
Classify news headlines into categories (World, Sports, Business, Sci/Tech) using a pretrained BERT model and deploy it for interactive use.

### Dataset Used
- Source: AG News dataset (via Hugging Face `datasets`)
- Features Used: Headline text
- Target Variable: Category label (0=World, 1=Sports, 2=Business, 3=Sci/Tech)

### Methodology / Approach
1. Load AG News dataset using `datasets`.
2. Explore and analyze class distribution.
3. Use BERT tokenizer (`bert-base-uncased`) to tokenize headlines with padding and truncation.
4. Fine-tune pretrained BERT for sequence classification with 4 output labels.
5. Train model using Hugging Face `Trainer` with evaluation at each epoch.
6. Compute metrics: Accuracy, Weighted F1 score, Confusion Matrix.
7. Visualize predictions using a confusion matrix.
8. Save trained model and tokenizer.
9. Deploy model with Streamlit for interactive headline classification with confidence scores.

### Key Results and Findings
- Model achieved high accuracy and F1-score on the test set.
- Confusion matrix shows that most misclassifications occur in similar categories (e.g., Sci/Tech vs Business headlines).
- Streamlit app allows real-time classification of new headlines.
- The project demonstrates an end-to-end NLP workflow: preprocessing → fine-tuning → evaluation → deployment.

**Note:** The model folder is not included in the repo due to large file size.  
To run locally, download the trained model and place it in `model/news_classifier_model/`.

### Notebook
Notebook: https://colab.research.google.com/drive/1BbINotWGM7nyYhiHYm3tPfkA-rHGex4O?usp=sharing
