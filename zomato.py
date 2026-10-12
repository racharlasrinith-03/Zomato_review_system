

import os
import joblib
import pandas as pd
import streamlit as st

# Get the absolute path of the current project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "zomato_sentiment_model.pkl"
)

# Page configuration
st.set_page_config(
    page_title="Zomato Review Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# Load the trained ML model
@st.cache_resource
def load_model():
    if not os.path.isfile("zomato_sentiment_model.pkl"):
        return None
    return joblib.load("zomato_sentiment_model.pkl")

model = load_model()

# Application header
st.title("🍽️ Zomato Review Sentiment Analyzer")
st.write(
    "Analyze restaurant reviews using Machine Learning "
    "and discover customer sentiment."
)

st.divider()

# Check model availability
if model is None:
    st.error(
        "Trained model not found. Please run train_model.py "
        "first to generate the model file."
    )
    st.code("python train_model.py")
    st.stop()

# Review input
st.subheader("📝 Analyze Customer Review")

review = st.text_area(
    "Enter your restaurant review",
    placeholder=(
        "Example: The food was delicious and "
        "the service was excellent..."
    ),
    height=150
)

if st.button("🔍 Analyze Review", type="primary"):
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        # Predict sentiment
        prediction = model.predict([review])[0]

        # Get probability estimates
        probabilities = model.predict_proba([review])[0]
        classes = model.classes_

        st.divider()
        st.subheader("📊 Sentiment Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == "Positive":
                st.success("😊 Positive Review")
            elif prediction == "Negative":
                st.error("😞 Negative Review")
            else:
                st.info("😐 Neutral Review")

        with col2:
            confidence = max(probabilities) * 100
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        st.caption(
            "Confidence represents the model's estimated probability, "
            "not a guarantee of prediction accuracy."
        )

        # Display sentiment probabilities
        results = pd.DataFrame({
            "Sentiment": classes,
            "Probability (%)": (
                probabilities * 100
            ).round(2)
        })

        st.subheader("📈 Sentiment Probability Breakdown")

        st.bar_chart(
            results.set_index("Sentiment")["Probability (%)"]
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

# Example reviews
st.divider()
st.subheader("💡 Sample Reviews")

examples = {
    "Positive": (
        "The food was amazing, fresh, and delicious. "
        "The staff were very friendly."
    ),
    "Negative": (
        "The food was cold and tasteless. "
        "The service was terrible."
    ),
    "Neutral": (
        "The food was average and the overall "
        "experience was okay."
    )
}

for sentiment, example in examples.items():
    with st.expander(f"{sentiment} example"):
        st.write(example)

st.divider()
st.caption(
    "Zomato Review Sentiment Analyzer | "
    "Machine Learning Project using TF-IDF and Logistic Regression"
)

