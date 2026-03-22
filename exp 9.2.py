print("S.SAMRITHA 24BAD103")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 9.1\ml-100k\u.data", sep='\t',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

ratings = ratings.drop('timestamp', axis=1)

item_user_matrix = ratings.pivot(index='movie_id',
                                 columns='user_id',
                                 values='rating')

item_user_filled = item_user_matrix.fillna(0)

item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity,
                                 index=item_user_matrix.index,
                                 columns=item_user_matrix.index)

def get_similar_items(movie_id, n=5):
    similar_items = item_similarity_df[movie_id].sort_values(ascending=False)
    return similar_items[1:n+1]

def predict_rating(user_id, movie_id):
    similar_items = get_similar_items(movie_id)
    sim_scores = similar_items.values
    user_ratings = item_user_filled.loc[similar_items.index, user_id]
    
    if np.sum(sim_scores) == 0:
        return 0
    
    return np.dot(user_ratings, sim_scores) / np.sum(sim_scores)

def recommend_items(user_id, n=5):
    user_data = ratings[ratings['user_id'] == user_id]
    watched_movies = user_data['movie_id'].tolist()
    
    predictions = {}
    
    for movie in item_user_matrix.index:
        if movie not in watched_movies:
            predictions[movie] = predict_rating(user_id, movie)
    
    recommended = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return recommended[:n]

actual = []
predicted = []

for user in item_user_matrix.columns:
    for movie in item_user_matrix.index:
        if not np.isnan(item_user_matrix.loc[movie, user]):
            pred = predict_rating(user, movie)
            actual.append(item_user_matrix.loc[movie, user])
            predicted.append(pred)

rmse = np.sqrt(mean_squared_error(actual, predicted))
print("RMSE:", rmse)

def precision_at_k(user_id, k=5):
    recs = recommend_items(user_id, k)
    recommended_items = [i[0] for i in recs]
    
    user_data = ratings[ratings['user_id'] == user_id]
    relevant_items = user_data[user_data['rating'] >= 4]['movie_id'].tolist()
    
    relevant_count = len(set(recommended_items) & set(relevant_items))
    return relevant_count / k

precision = precision_at_k(1, 5)
print("Precision@5:", precision)

user_id = 1
recommendations = recommend_items(user_id, 5)

print(f"\nTop 5 Recommendations for User {user_id}:")
for movie, rating in recommendations:
    print(f"Movie ID: {movie}, Predicted Rating: {rating:.2f}")

movie_id = 1
similar_items = get_similar_items(movie_id, 5)

print(f"\nTop Similar Items for Movie {movie_id}:")
for movie, score in similar_items.items():
    print(f"Movie ID: {movie}, Similarity: {score:.2f}")

plt.figure(figsize=(10,8))
sns.heatmap(item_similarity_df.iloc[:25, :25])
plt.title("Item Similarity Matrix")
plt.xlabel("Items")
plt.ylabel("Items")
plt.show()

movies = [str(i) for i in similar_items.index]
scores = similar_items.values

plt.figure(figsize=(8,5))
plt.bar(movies, scores)
plt.title(f"Top Similar Items for Movie {movie_id}")
plt.xlabel("Movie ID")
plt.ylabel("Similarity Score")
plt.show()

user_recs = [i[1] for i in recommendations]

plt.figure(figsize=(8,5))
plt.plot(range(1, len(user_recs)+1), user_recs, marker='o')
plt.title("Recommendation Scores (Item-Based)")
plt.xlabel("Rank")
plt.ylabel("Score")
plt.show()
