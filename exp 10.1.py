print("S.SAMRITHA 24BAD103")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds

# Load dataset
ratings = pd.read_csv(r"C:\Users\Lenovo\Desktop\u.data", sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv(r"C:\Users\Lenovo\Downloads\u.item", sep='|',
                     encoding='latin-1', names=['item_id', 'title'], usecols=[0, 1])
# Preprocess
data = pd.merge(ratings, movies, on='item_id').drop('timestamp', axis=1)

# User-item matrix
user_item_matrix = data.pivot_table(index='user_id',
                                    columns='title',
                                    values='rating',
                                    aggfunc='mean')
user_item_matrix_filled = user_item_matrix.fillna(0)

# Normalize
user_mean = np.mean(user_item_matrix_filled, axis=1)
matrix_demeaned = user_item_matrix_filled.sub(user_mean, axis=0)

# SVD
k = 50
U, sigma, Vt = svds(matrix_demeaned.values, k=k)
sigma = np.diag(sigma)

# Reconstruction
reconstructed = np.dot(np.dot(U, sigma), Vt) + user_mean.values.reshape(-1, 1)
predicted_ratings = pd.DataFrame(reconstructed,
                                 columns=user_item_matrix.columns,
                                 index=user_item_matrix.index)
# Evaluation
actual = user_item_matrix.values
predicted = predicted_ratings.values
mask = actual > 0

rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
mae = mean_absolute_error(actual[mask], predicted[mask])

print("RMSE:", rmse)
print("MAE:", mae)

# Recommendation function
def recommend_movies(user_id, n=5):
    user_pred = predicted_ratings.loc[user_id].sort_values(ascending=False)
    rated = user_item_matrix.loc[user_id].dropna().index
    return user_pred.drop(rated).head(n)

# Top recommendations
top_movies = recommend_movies(1, 5)
print("\nTop Recommended Movies:\n", top_movies)

# Heatmap (Original matrix)
plt.figure(figsize=(10,4))
sns.heatmap(user_item_matrix_filled.iloc[:20, :20])
plt.title("Original Matrix")
plt.show()

# Heatmap (Reconstructed matrix)
plt.figure(figsize=(10,4))
sns.heatmap(predicted_ratings.iloc[:20, :20])
plt.title("Reconstructed Matrix")
plt.show()

# Error vs latent factors (RMSE & MAE)
k_values = [10, 20, 30, 40, 50]
rmse_list = []
mae_list = []

for k in k_values:
    U, sigma, Vt = svds(matrix_demeaned.values, k=k)
    sigma = np.diag(sigma)
    
    recon = np.dot(np.dot(U, sigma), Vt) + user_mean.values.reshape(-1, 1)
    
    rmse_k = np.sqrt(mean_squared_error(actual[mask], recon[mask]))
    mae_k = mean_absolute_error(actual[mask], recon[mask])
    
    rmse_list.append(rmse_k)
    mae_list.append(mae_k)

plt.figure()
plt.plot(k_values, rmse_list, marker='o', label='RMSE')
plt.plot(k_values, mae_list, marker='s', label='MAE')
plt.xlabel("Latent Factors (k)")
plt.ylabel("Error")
plt.title("RMSE & MAE vs Latent Factors")
plt.legend()
plt.show()
# Top recommended movies visualization
plt.figure()
top_movies.plot(kind='bar')
plt.xlabel("Movies")
plt.ylabel("Predicted Rating")
plt.title("Top Recommended Movies for User 1")
plt.xticks(rotation=60, ha='right')
plt.show()
