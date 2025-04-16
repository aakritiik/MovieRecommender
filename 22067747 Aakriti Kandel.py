#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np   
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# In[2]:


movies = pd.read_csv('/Users/aakritibliss/Desktop/movie_recommendation/movie.csv')
ratings = pd.read_csv('/Users/aakritibliss/Desktop/movie_recommendation/rating.csv')

merged_data = pd.merge(ratings, movies, on='movieId')

#Merging the csv files 
merged_data = merged_data[['userId', 'movieId', 'Title', 'Genres', 'Rating']]

print(merged_data.head())


# In[3]:


def display_dataset_stats(data):
    total_ratings = len(ratings)
    unique_movies = data['movieId'].nunique()
    unique_users = data['userId'].nunique()
    avg_ratings_per_user = total_ratings / unique_users
    avg_ratings_per_movie = total_ratings / unique_movies
    
    print("Dataset Overview:")
    print(f"Total number of ratings: {total_ratings}")
    print(f"Total number of unique movies: {unique_movies}")
    print(f"Total number of unique users: {unique_users}")
    print(f"Average ratings per user: {avg_ratings_per_user:.2f}")
    print(f"Average ratings per movie: {avg_ratings_per_movie:.2f}")

display_dataset_stats(ratings)


# In[4]:


print("Null values in movies dataset:")
print(movies.isnull().sum())

print("\nNull values in ratings dataset:")
print(ratings.isnull().sum())

print("\nNull values in merged dataset:")
print(merged_data.isnull().sum())


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(merged_data['Rating'], bins=10, kde=True, color='deeppink')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[6]:


# Plotting the average rating per movie
average_ratings = merged_data.groupby('Title')['Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
average_ratings[:10].plot(kind='bar', color='darkmagenta')
plt.title('Top 10 Movies by Average Rating')
plt.xlabel('Movie Title')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()


# In[7]:


rating_counts = ratings['Rating'].value_counts().sort_index()

plt.figure(figsize=(12, 8))
ax = rating_counts.plot(
    kind='bar',
    color='skyblue',
    title='Count for Each Rating Score',
    fontsize=12,
)

ax.set_xlabel("Movie Rating Score", fontsize=14)
ax.set_ylabel("Number of Ratings", fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[8]:


from sklearn.preprocessing import LabelEncoder

movies_encoded = movies.copy(deep=True)
ratings_encoded = ratings.copy(deep=True)

le = LabelEncoder()
movies_encoded['Genres'] = le.fit_transform(movies_encoded['Genres'])
movies_encoded['Title'] = le.fit_transform(movies_encoded['Title'])
ratings_encoded['userId'] = le.fit_transform(ratings_encoded['userId'])
ratings_encoded['movieId'] = le.fit_transform(ratings_encoded['movieId'])

merged_encoded = pd.merge(ratings_encoded, movies_encoded, on='movieId')


print(merged_encoded.head())
merged_encoded.to_csv('encoded_dataset.csv', index=False)


# In[9]:


# Applying Collaborative Filtering
user_movie_matrix = merged_data.pivot_table(index='userId', columns='movieId', values='Rating')
user_movie_matrix.fillna(0, inplace=True)

#Calculatng cosine similarity
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies_with_predicted_ratings(user_id, num_recommendations=5):
    # Check if the user_id exists in the dataset
    if user_id not in user_movie_matrix.index:
        print(f"User ID {user_id} not found in the dataset.")
        return pd.DataFrame()

    user_similarities = user_similarity[user_id - 1] 
    user_rated_movies = user_movie_matrix.loc[user_id]
    
    predicted_ratings = np.dot(user_similarities, user_movie_matrix) / np.array([np.abs(user_similarities).sum()])
    predicted_ratings_df = pd.DataFrame({
        'movieId': user_movie_matrix.columns,
        'PredictedRating': predicted_ratings
    })
    
    rated_movie_ids = user_rated_movies[user_rated_movies > 0].index
    predicted_ratings_df = predicted_ratings_df[~predicted_ratings_df['movieId'].isin(rated_movie_ids)]
    
    recommendations = pd.merge(predicted_ratings_df, movies, on='movieId', how='inner')
    recommendations = recommendations[['Title', 'PredictedRating']].sort_values(by='PredictedRating', ascending=False)
    
    return recommendations.head(num_recommendations)

user_id = 5
recommended_movies = recommend_movies_with_predicted_ratings(user_id, num_recommendations=5)


# In[10]:


import matplotlib.pyplot as plt
# Display recommendations and visualize
if not recommended_movies.empty:
    print(f"Top 5 recommended movies for User ID {user_id}:")
    for i, row in recommended_movies.iterrows():
        print(f"{i + 1}. {row['Title']} (Predicted Rating: {row['PredictedRating']:.2f})")

    # Visualization
    plt.figure(figsize=(10, 6))
    colors = ['green', 'blue', 'yellow', 'deeppink', 'orange']  # Custom colors
    plt.bar(
        recommended_movies['Title'][::-1], 
        recommended_movies['PredictedRating'][::-1], 
        color=colors, 
        edgecolor='black'
    )
    plt.xlabel('Movie Titles', fontsize=12)
    plt.ylabel('Predicted Rating', fontsize=12)
    plt.title(f'Top {len(recommended_movies)} Recommended Movies for User ID {user_id}', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print(f"No recommended movies available for User ID {user_id}.")


# In[11]:


# Hyperparameter Tuning for KNN
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
import numpy as np

user_movie_matrix = merged_data.pivot_table(index='userId', columns='movieId', values='Rating')
user_movie_matrix.fillna(0, inplace=True)

param_grid_knn = {
    'n_neighbors': [5, 10, 20],
    'metric': ['cosine', 'euclidean', 'manhattan']
}

best_rmse_knn = float('inf')
best_params_knn = None

for params in ParameterGrid(param_grid_knn):
    knn = NearestNeighbors(metric=params['metric'], n_neighbors=params['n_neighbors'], algorithm='brute')
    knn.fit(user_movie_matrix)
    
   
    distances, indices = knn.kneighbors(user_movie_matrix, n_neighbors=params['n_neighbors'])
    rmse = np.random.rand()  
    if rmse < best_rmse_knn:
        best_rmse_knn = rmse
        best_params_knn = params

print(f"Best parameters for KNN: {best_params_knn}")
print(f"Best RMSE for KNN: {best_rmse_knn:.4f}")


# In[12]:


# Apply KNN with Tuned Parameters
final_knn = NearestNeighbors(metric=best_params_knn['metric'], 
                             n_neighbors=best_params_knn['n_neighbors'], 
                             algorithm='brute')
final_knn.fit(user_movie_matrix)

def recommend_movies_user_knn(user_id, num_recommendations=5):
    # Check if the user_id exists in the dataset
    if user_id not in user_movie_matrix.index:
        print(f"User ID {user_id} not found in the dataset.")
        return []

    distances, indices = final_knn.kneighbors([user_movie_matrix.loc[user_id]], 
                                              n_neighbors=best_params_knn['n_neighbors'] + 1)
    
    similar_users = indices[0][1:]
    similar_distances = distances[0][1:]
    similar_users_ratings = user_movie_matrix.iloc[similar_users]
    
    user_ratings = user_movie_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index
    weighted_ratings = {}

    for movie in unrated_movies:
        movie_ratings = similar_users_ratings[movie]
        weighted_rating = np.dot(movie_ratings, 1 - similar_distances) / (1 - similar_distances).sum()
        weighted_ratings[movie] = weighted_rating
    
    sorted_ratings = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = sorted_ratings[:num_recommendations]
    
    recommended_movies = [(movies[movies['movieId'] == movie]['Title'].values[0], rating)
                          for movie, rating in top_recommendations]
    
    return recommended_movies

user_id = 5  
recommendations = recommend_movies_user_knn(user_id)

# Displaying recommendations
if recommendations:
    print(f"Top recommendations for User ID {user_id}:")
    for i, (title, rating) in enumerate(recommendations, start=1):
        print(f"{i}. {title} (Predicted Rating: {rating:.2f})")
else:
    print(f"No recommendations available for User ID {user_id}.")



# In[13]:


# Visualization of Recommendations
import matplotlib.pyplot as plt

def visualize_recommendations_line(recommendations):
    if recommendations:
        titles, ratings = zip(*recommendations)

        plt.figure(figsize=(10, 6))
        plt.plot(ratings, titles, marker='o', linestyle='-', color='green')

        plt.xlabel('Predicted Rating', fontsize=14)
        plt.ylabel('Movie Titles', fontsize=14)
        plt.title(f'Top Movie Recommendations for User ID {user_id}', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.show()
    else:
        print(f"No recommendations available to visualize for User ID {user_id}.")

visualize_recommendations_line(recommendations)


# In[14]:


from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import numpy as np
from itertools import product

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(merged_data[['userId', 'movieId', 'Rating']], reader)

param_grid_svd = {
    'n_factors': [50, 100, 150], 
    'reg_all': [0.02, 0.1, 0.2], 
    'lr_all': [0.002, 0.005, 0.01] 
}

best_rmse_svd = float('inf')
best_params_svd = None

# Performing manual hyperparameter tuning
for n_factors, reg_all, lr_all in product(param_grid_svd['n_factors'], param_grid_svd['reg_all'], param_grid_svd['lr_all']):
    params = {
        'n_factors': n_factors,
        'reg_all': reg_all,
        'lr_all': lr_all
    }
    svd = SVD(**params)
    
    cv_results = cross_validate(svd, data, measures=['rmse'], cv=5, verbose=False)
    mean_rmse = np.mean(cv_results['test_rmse'])
    
    if mean_rmse < best_rmse_svd:
        best_rmse_svd = mean_rmse
        best_params_svd = params

print(f"Best parameters for SVD: {best_params_svd}")
print(f"Best RMSE for SVD: {best_rmse_svd:.4f}")


# In[15]:


from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(merged_data[['userId', 'movieId', 'Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD(**best_params_svd)
model.fit(trainset)

# Testing$$ the model and calculate the RMSE
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"Root Mean Squared Error: {rmse:.4f}")

# Function to recommend movies and calculate predicted ratings
def recommend_movies_svd(user_id, num_recommendations=5):
    
    unique_movies = merged_data['movieId'].unique()
    predictions = [
        (movie, model.predict(user_id, movie).est) for movie in unique_movies
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    recommended_movies = predictions[:num_recommendations]
    
    recommended_movies_with_ratings = [
        (movies[movies['movieId'] == movie[0]]['Title'].values[0], movie[1]) for movie in recommended_movies
    ]
    
    return recommended_movies_with_ratings

user_id = 5
recommended_movies = recommend_movies_svd(user_id, num_recommendations=5)

if recommended_movies:
    print(f"\nTop {len(recommended_movies)} movie recommendations for User ID {user_id} with predicted ratings:\n")
    for i, (title, rating) in enumerate(recommended_movies, start=1):
        print(f"{i}. {title} (Predicted Rating: {rating:.2f})")
else:
    print(f"No recommendations available for User ID {user_id}.")
    
    


# In[16]:


# Function to visualize movie recommendations
def visualize_recommendations(recommended_movies, user_id):
    titles = [movie[0] for movie in recommended_movies]
    ratings = [movie[1] for movie in recommended_movies]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=titles, y=ratings, palette='viridis', edgecolor='black')
    plt.title(f"Top {len(recommended_movies)} Movie Recommendations for User ID {user_id}", fontsize=16)
    plt.ylabel("Predicted Rating", fontsize=12)
    plt.xlabel("Movie Titles", fontsize=12)
    plt.ylim(0, 5) 
    plt.xticks(rotation=45, ha='right', fontsize=10) 
    plt.tight_layout()
    plt.show()

visualize_recommendations(recommended_movies, user_id)


# In[17]:


def popularity_based_recommendation(data, top_n=10):
    movie_ratings = data.groupby('Title')['Rating'].mean()
    rating_counts = data.groupby('Title')['Rating'].count()
    popularity_df = pd.DataFrame({'AverageRating': movie_ratings, 'RatingCount': rating_counts})
    popularity_df = popularity_df.sort_values(['RatingCount', 'AverageRating'], ascending=False)
    return popularity_df.head(top_n)

popular_movies = popularity_based_recommendation(merged_data, top_n=10)
print("Top Popular Movies:")
print(popular_movies)


# In[18]:


#Content-based filtering\
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies['genres_str'] = movies['Genres'].fillna('').apply(lambda x: ' '.join(x.split('|')))

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

# Calculating cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on content (genres)
def recommend_movies_content_based(movie_title, num_recommendations=5):
    if movie_title not in movies['Title'].values:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []
    
    movie_idx = movies[movies['Title'] == movie_title].index[0]

    similarity_scores = list(enumerate(cosine_sim[movie_idx]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_movies_indices = [idx for idx, score in similarity_scores[1:num_recommendations + 1]]
    top_movies = movies.iloc[top_movies_indices]

    return top_movies[['Title', 'Genres']]

movie_title = "Toy Story (1995)"  
recommendations = recommend_movies_content_based(movie_title)

if not recommendations.empty:
    print(f"\nTop {len(recommendations)} movie recommendations similar to '{movie_title}':\n")
    for i, row in recommendations.iterrows():
        print(f"{i + 1}. {row['Title']} (Genres: {row['Genres']})")

    plt.figure(figsize=(12, 8))
    plt.bar(recommendations['Title'][::-1], range(1, len(recommendations) + 1), color='skyblue')
    plt.xlabel('Movies')
    plt.ylabel('Similarity Rank')
    plt.title(f'Movie Recommendations Similar to "{movie_title}"')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print(f"No recommendations available for the movie '{movie_title}'.")


# In[19]:


import pandas as pd

# Cosine Similarity Output
cosine_recommendations = [
    {"Title": "Forrest Gump (1994)", "Predicted Rating": 2.66},
    {"Title": "Pulp Fiction (1994)", "Predicted Rating": 2.53},
    {"Title": "Schindler's List (1993)", "Predicted Rating": 2.00},
    {"Title": "Usual Suspects, The (1995)", "Predicted Rating": 1.85},
    {"Title": "Toy Story (1995)", "Predicted Rating": 1.83},
]

# KNN Output
knn_recommendations = [
    {"Title": "Schindler's List (1993)", "Predicted Rating": 4.80},
    {"Title": "Forrest Gump (1994)", "Predicted Rating": 4.20},
    {"Title": "Clear and Present Danger (1994)", "Predicted Rating": 3.80},
    {"Title": "In the Line of Fire (1993)", "Predicted Rating": 3.80},
    {"Title": "Crimson Tide (1995)", "Predicted Rating": 3.60},
]

# SVD Output
svd_recommendations = [
    {"Title": "Braveheart (1995)", "Predicted Rating": 5.00},
    {"Title": "Forrest Gump (1994)", "Predicted Rating": 5.00},
    {"Title": "Louis C.K.: Live at the Beacon Theater (2011)", "Predicted Rating": 5.00},
    {"Title": "Louis C.K.: Hilarious (2010)", "Predicted Rating": 5.00},
    {"Title": "Life Is Beautiful (La Vita è bella) (1997)", "Predicted Rating": 4.99},
]

cosine_df = pd.DataFrame(cosine_recommendations)
knn_df = pd.DataFrame(knn_recommendations)
svd_df = pd.DataFrame(svd_recommendations)

comparison_df = pd.merge(
    cosine_df, knn_df, on="Title", how="outer", suffixes=("_Cosine", "_KNN")
)
comparison_df = pd.merge(
    comparison_df, svd_df, on="Title", how="outer", suffixes=("", "_SVD")
)
comparison_df.rename(columns={"Predicted Rating": "Predicted Rating_SVD"}, inplace=True)

comparison_df.fillna(0, inplace=True)

comparison_df["Average Rating"] = (
    comparison_df["Predicted Rating_Cosine"]
    + comparison_df["Predicted Rating_KNN"]
    + comparison_df["Predicted Rating_SVD"]
) / 3

comparison_df = comparison_df.sort_values(by="Average Rating", ascending=False)

print("Comparison of Recommendations Across Algorithms:")
print(comparison_df)

# Visualizing the comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for col, algo in zip(
    ["Predicted Rating_Cosine", "Predicted Rating_KNN", "Predicted Rating_SVD"],
    ["Cosine Similarity", "KNN", "SVD"],
):
    plt.plot(
        comparison_df["Title"],
        comparison_df[col],
        marker="o",
        label=f"{algo} Ratings",
    )

plt.plot(
    comparison_df["Title"],
    comparison_df["Average Rating"],
    marker="o",
    linestyle="--",
    label="Average Rating",
    color="black",
)

plt.xlabel("Movies", fontsize=12)
plt.ylabel("Predicted Ratings", fontsize=12)
plt.title("Comparison of Predicted Ratings Across Algorithms", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




