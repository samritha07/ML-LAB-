print("S.SAMRITHA 24BAD103")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv(r"C:\Users\Lenovo\Downloads\Exp 9.1\ml-100k\u.data", sep='\t',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

ratings = ratings.drop('timestamp', axis=1)

user_item_matrix = ratings.pivot(index='user_id',
                                 columns='movie_id',
                                 values='rating')

user_item_filled = user_item_matrix.fillna(0)

user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_item_matrix.index,
                                 columns=user_item_matrix.index)

def get_similar_users(user_id, n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    return similar_users[1:n+1]

def predict_rating(user_id, movie_id):
    similar_users = get_similar_users(user_id)
    sim_scores = similar_users.values
    ratings_local = user_item_filled.loc[similar_users.index, movie_id]
    if np.sum(sim_scores) == 0:
        return 0
    return np.dot(ratings_local, sim_scores) / np.sum(sim_scores)

def recommend_movies(user_id, n=5):
    user_data = user_item_matrix.loc[user_id]
    unseen_movies = user_data[user_data.isna()].index
    predictions = {}
    for movie in unseen_movies:
        predictions[movie] = predict_rating(user_id, movie)
    recommended = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return recommended[:n]

actual = []
predicted = []

for user in user_item_matrix.index:
    for movie in user_item_matrix.columns:
        if not np.isnan(user_item_matrix.loc[user, movie]):
            pred = predict_rating(user, movie)
            actual.append(user_item_matrix.loc[user, movie])
            predicted.append(pred)

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print("RMSE:", rmse)
print("MAE:", mae)

user_id = 1
recommendations = recommend_movies(user_id, 5)

print(f"\nTop 5 Recommendations for User {user_id}:")
for movie, rating in recommendations:
    print(f"Movie ID: {movie}, Predicted Rating: {rating:.2f}")

plt.figure(figsize=(10,8))
sns.heatmap(user_item_filled.iloc[:25, :25])
plt.title("User-Item Interaction Matrix (Sample)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(user_similarity_df.iloc[:25, :25])
plt.title("User Similarity Matrix (Cosine Similarity)")
plt.xlabel("Users")
plt.ylabel("Users")
plt.show()

movies = [str(i[0]) for i in recommendations]
scores = [i[1] for i in recommendations]

plt.figure(figsize=(8,5))
plt.bar(movies, scores)
plt.title(f"Top Recommended Movies for User {user_id}")
plt.xlabel("Movie ID")
plt.ylabel("Predicted Rating")
plt.show()

plt.figure(figsize=(6,4))
ratings['rating'].hist(bins=5)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

sparsity = 1.0 - (np.count_nonzero(user_item_matrix) / user_item_matrix.size)
print(f"Sparsity of User-Item Matrix: {sparsity:.2f}")

plt.figure(figsize=(6,4))
plt.imshow(user_item_matrix.isna(), aspect='auto')
plt.title("Sparsity Pattern (White = Missing)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()
