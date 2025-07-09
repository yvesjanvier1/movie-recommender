
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Use relative paths so it works on Streamlit Cloud / GitHub
model = joblib.load('svd_recommender.pkl')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

movies['genres'] = movies['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_top_n_recommendations(user_id, n=10):
    rated = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movie_ids = movies['movieId'].unique()
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated:
            pred = model.predict(uid=user_id, iid=movie_id)
            predictions.append((movie_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    return [(movies[movies['movieId'] == mid]['title'].values[0], round(r, 2)) for mid, r in top_n]

def content_based_recommend(title):
    if title not in indices:
        return f"'{title}' not found."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:11]]
    return movies.iloc[sim_indices][['title', 'genres']]

def hybrid_recommend(user_id, title):
    if title not in indices:
        return f"'{title}' not found."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:21]]
    cbf_movies = movies.iloc[sim_indices]
    preds = []
    for _, row in cbf_movies.iterrows():
        pred = model.predict(uid=user_id, iid=row['movieId'])
        preds.append((row['title'], round(pred.est, 2)))
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:10]

st.title("🎬 Movie Recommender System")
user_id = st.number_input("Enter User ID", min_value=1, step=1)
movie_title = st.selectbox("Select a Movie", options=movies['title'].unique())

if st.button("Recommend"):
    st.subheader("🔁 Collaborative Recommendations")
    st.write(get_top_n_recommendations(user_id))

    st.subheader("🎯 Content-Based Recommendations")
    st.write(content_based_recommend(movie_title))

    st.subheader("💡 Hybrid Recommendations")
    st.write(hybrid_recommend(user_id, movie_title))
