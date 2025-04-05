# You can recreate the environment later using:
conda env create -f environment.yml
# you can only install requirments into your repository using:
pip install -r requirements.txt.

# Hybrid_Movie_Recommendatin_System

1️⃣ Collaborative Filtering (SVD)
Use Singular Value Decomposition (SVD) to decompose the user-item matrix and generate predictions.

Train the model using the ratings.csv file.

Generate recommendations for users based on learned embeddings.

2️⃣ Content-Based Filtering (Deep Learning)
Use movie metadata (movies.csv, tags.csv) to train a neural network.

Convert genres, titles, and tags into numerical embeddings (e.g., TF-IDF, Word2Vec, or Transformers).

Train a deep learning model to predict a user’s preference for a given movie.

3️⃣ Ranking Model
Take outputs from both SVD and Deep Learning models.

Combine them using a meta-ranking model (e.g., XGBoost, Neural Network, or Logistic Regression).

Train the ranking model using interactions between the two recommendation scores and the actual user ratings.


https://github.com/user-attachments/assets/5ec13540-c0fc-4f04-8f93-3383729ba98b

