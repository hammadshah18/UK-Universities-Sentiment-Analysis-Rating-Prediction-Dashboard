import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ==============================
# Load Data & Models
# ==============================
@st.cache_data
def load_data():
    file_id = "1e5x19deGmIyyCdUNPXTf2UAY2zMtuAnz"  # replace with your file ID
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    

    
    df = pd.read_csv(url)
    df["review_length"] = df["Review"].apply(lambda x: len(str(x).split()))

    return df

df = load_data()

# Load vectorizer + encoder + models
with open("vector1.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("encode.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("predict1.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("predict2.pkl", "rb") as f:
    rating_model = pickle.load(f)

# ==============================
# Dashboard Title
# ==============================
st.title("🎓 University Review Sentiment & Ranking Dashboard")


# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Choose an option:",
    [
        "📑 View Dataset",
         
        "🏆 University Ranking", 

        "📊 Visualizations",

        "😊 Sentiment Prediction",
        
        "⭐ Rating Prediction"
    ]
)

# ==============================
# Dataset Viewer
# ==============================
if page == "📑 View Dataset":
    st.subheader("📑 Original Dataset")
    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head(50))  # Show first 50 rows

    st.download_button(
        label="📥 Download Full Dataset as CSV",
        data=df.to_csv(index=False),
        file_name="university_reviews.csv",
        mime="text/csv"
    )

# ==============================
# University Ranking
# ==============================
elif page == "🏆 University Ranking":
    st.subheader("🏆 University Ranking")

    uni_rank = df.groupby("University").agg({
        "Rating": "mean",
        "Sentiment": lambda x: (x == "positive").mean() * 100
    }).reset_index()

    uni_rank = uni_rank.rename(columns={"Rating": "Avg Rating", "Sentiment": "% Positive"})
    uni_rank = uni_rank.sort_values(by="Avg Rating", ascending=False)

    st.dataframe(uni_rank.head(20))

# ==============================
# Visualizations (EDA)
# ==============================
elif page == "📊 Visualizations":
    st.subheader("📊 Data Visualizations")

    graph_option = st.sidebar.selectbox(
        "Select a graph to display:",
        [
            "Sentiment Distribution",
            "Average Rating Distribution",
            "Top 10 Universities by Avg Rating",
            "Top 10 Universities by % Positive Sentiment",
            "City-wise Avg Rating",
            "Rating vs Sentiment Relationship",
            "Review Length Distribution",
            "City-wise Sentiment Distribution",
            "Correlation Heatmap",
            "Word Cloud of Reviews"
        ]
    )

    if graph_option == "Sentiment Distribution":
        fig, ax = plt.subplots()
        sns.countplot(x="Sentiment", data=df, ax=ax, palette="Set2")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    elif graph_option == "Average Rating Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df["Rating"], bins=5, kde=True, ax=ax, color="skyblue")
        ax.set_title("Average Rating Distribution")
        st.pyplot(fig)

    elif graph_option == "Top 10 Universities by Avg Rating":
        uni_rank = df.groupby("University")["Rating"].mean().reset_index().sort_values("Rating", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x="Rating", y="University", data=uni_rank.head(10), ax=ax, palette="Blues_r")
        ax.set_title("Top 10 Universities by Avg Rating")
        st.pyplot(fig)

    elif graph_option == "Top 10 Universities by % Positive Sentiment":
        uni_rank = df.groupby("University")["Sentiment"].apply(lambda x: (x == "positive").mean() * 100).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x="Sentiment", y="University", data=uni_rank.sort_values("Sentiment", ascending=False).head(10), ax=ax, palette="Greens_r")
        ax.set_title("Top 10 Universities by Positive Sentiment")
        st.pyplot(fig)

    elif graph_option == "City-wise Avg Rating":
        city_avg = df.groupby("City")["Rating"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=city_avg.values, y=city_avg.index, ax=ax, palette="coolwarm")
        ax.set_title("Top Cities by Average Rating")
        st.pyplot(fig)

    elif graph_option == "Rating vs Sentiment Relationship":
        fig, ax = plt.subplots()
        sns.boxplot(x="Sentiment", y="Rating", data=df, ax=ax, palette="Set3")
        ax.set_title("Rating vs Sentiment")
        st.pyplot(fig)

    elif graph_option == "Review Length Distribution":
        df["review_length"] = df["Review"].apply(lambda x: len(str(x).split()))
        fig, ax = plt.subplots()
        sns.histplot(df["review_length"], bins=50, kde=True, ax=ax, color="purple")
        ax.set_title("Review Length Distribution")
        st.pyplot(fig)

    elif graph_option == "City-wise Sentiment Distribution":
        city_sent = df.groupby(["City", "Sentiment"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="City", y="count", hue="Sentiment", data=city_sent, ax=ax)
        ax.set_title("City-wise Sentiment Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # elif graph_option == "Monthly Review Trends":
    #     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    #     monthly = df.groupby(df["Date"].dt.to_period("M"))["Rating"].mean().reset_index()
    #     monthly["Date"] = monthly["Date"].astype(str)
    #     fig, ax = plt.subplots()
    #     sns.lineplot(x="Date", y="Rating", data=monthly, marker="o", ax=ax, color="blue")
    #     ax.set_title("Monthly Review Trends")
    #     plt.xticks(rotation=45)
    #     st.pyplot(fig)

    elif graph_option == "Correlation Heatmap":
        corr = df[["Rating", "review_length"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif graph_option == "Word Cloud of Reviews":
        text = " ".join(df["Review"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ==============================
# Sentiment Prediction
# ==============================
elif page == "😊 Sentiment Prediction":
    st.subheader("😊 Sentiment Prediction")

    user_review = st.text_area("Enter a student review:")

    if st.button("Predict Sentiment"):
        X_text = vectorizer.transform([user_review])
        sentiment_pred = sentiment_model.predict(X_text)[0]

        # Map 0/1/2 → labels
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_label = label_map.get(sentiment_pred, "Unknown")

        st.success(f"**Predicted Sentiment:** {sentiment_label}")



# ==============================
# Rating Prediction
# ==============================
elif page == "⭐ Rating Prediction":
    st.subheader("⭐ Rating Prediction")

    # user_review = st.text_area("Enter a student review:")
    user_uni = st.selectbox("Select University:", df["University"].unique())
    user_city = st.selectbox("Select City:", df["City"].unique())

    if st.button("Predict Rating"):
        # X_text = vectorizer.transform([user_review])
        X_cats = encoder.transform([[user_uni, user_city]])
        # X_final = hstack([X_text, X_cats])
        X_final = hstack([ X_cats])
        rating_pred = rating_model.predict(X_final)[0]

        st.success(f"**Predicted Rating:** {round(rating_pred, 2)} ⭐")




