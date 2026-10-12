# Zomato Review Sentiment Analyzer

A Machine Learning web application that analyzes restaurant reviews and predicts whether the sentiment is **Positive**, **Negative**, or **Neutral**.

## Features

- Enter a restaurant review and predict its sentiment.
- View the model's estimated confidence.
- Explore a probability breakdown for each sentiment.
- Use a simple web interface built with Streamlit.
- Apply TF-IDF text features and a Logistic Regression classifier.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Joblib

## Project Structure

```text
Zomato_Review_System/
├── app.py
├── train_model.py
├── requirements.txt
└── models/
    └── zomato_sentiment_model.pkl
```

The model file is generated when you run `train_model.py`. Do not expect it to exist before training.

## Installation

1. Open a terminal in the project folder.
2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

If you do not have a `requirements.txt` file yet, create one containing:

```text
pandas
scikit-learn
streamlit
joblib
```

## Train the Model

Run this command from the project folder:

```bash
python train_model.py
```

The script trains the model and saves it to:

```text
models/zomato_sentiment_model.pkl
```

The current training script uses a small demonstration dataset. Its test accuracy is not a reliable estimate of real-world performance.

## Run the Application

After training completes, start Streamlit:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Enter a restaurant review in the text box.
2. Click **Analyze Review**.
3. View the predicted sentiment, estimated confidence, and probability chart.

Example reviews:

- **Positive:** `The food was delicious and the service was excellent.`
- **Negative:** `The food was cold and the service was terrible.`
- **Neutral:** `The food was average and the overall experience was okay.`

The only input required by the current app is review text.

## Model Workflow

1. Collect labeled restaurant reviews.
2. Split the data into training and testing sets.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classifier.
5. Predict sentiment for a new review.
6. Display the result in Streamlit.

## Limitations

- The sample dataset is small and intended only to demonstrate the workflow.
- Predictions may be inaccurate for unfamiliar wording, sarcasm, mixed sentiment, or languages not represented in training data.
- Model confidence is an estimated probability, not a guarantee of correctness.
- For a more reliable project, train and evaluate using a larger, real, labeled restaurant-review dataset.

## Future Enhancements

- Use a real Zomato or restaurant review dataset.
- Add CSV upload for batch review analysis.
- Show overall sentiment statistics and charts.
- Add restaurant ratings and review filtering.
- Improve evaluation using precision, recall, F1-score, and a confusion matrix.

## Author

Machine Learning Project — Zomato Review Sentiment Analysis
