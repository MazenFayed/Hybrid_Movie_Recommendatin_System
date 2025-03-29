import tkinter as tk
from tkinter import messagebox, scrolledtext
import joblib  
import numpy as np  
import torch
import torch.nn as nn
import pandas as pd
import requests

# Define the MovieRecommender class (unchanged)
class MovieRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MovieRecommender, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Load data
movies = pd.read_csv(r"E:\Courses\recommendation system\project\Project_GUI\Data\movies.csv")
ratings = pd.read_csv(r"E:\Courses\recommendation system\project\Project_GUI\Data\ratings.csv")
links = pd.read_csv(r"E:\Courses\recommendation system\project\Project_GUI\Data\links.csv")
svd_model = joblib.load(r"E:\Courses\recommendation system\project\Project_GUI\Data\svd_model.pkl")
content_based_model = joblib.load(r"E:\Courses\recommendation system\project\Project_GUI\Data\content_based_model.pkl")
user_encoder = joblib.load(r"E:\Courses\recommendation system\project\Project_GUI\Data\user_encoder.pkl")
movie_encoder = joblib.load(r"E:\Courses\recommendation system\project\Project_GUI\Data\movie_encoder.pkl")
scaler = joblib.load(r"E:\Courses\recommendation system\project\Project_GUI\Data\scaler.pkl")

class RecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation System")
        self.root.geometry("1300x500")  # Increased width for 3 columns

        # User ID input
        self.user_label = tk.Label(root, text="Enter User ID:")
        self.user_label.grid(row=0, column=0, padx=10, pady=5)

        self.user_id_entry = tk.Entry(root)
        self.user_id_entry.grid(row=0, column=1, padx=10, pady=5)

        # Number of recommendations input
        self.recommendations_label = tk.Label(root, text="Number of Recommendations:")
        self.recommendations_label.grid(row=1, column=0, padx=10, pady=5)

        self.recommendations_entry = tk.Entry(root)
        self.recommendations_entry.grid(row=1, column=1, padx=10, pady=5)

        # Submit button
        self.submit_button = tk.Button(root, text="Get Recommendations", command=self.get_recommendations)
        self.submit_button.grid(row=2, columnspan=2, pady=10)

        # Display frame
        self.display_frame = tk.Frame(root)
        self.display_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        # Watched Movies Label
        self.watched_label = tk.Label(self.display_frame, text="Watched Movies", font=("Arial", 12, "bold"))
        self.watched_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.watched_movies_text = scrolledtext.ScrolledText(self.display_frame, width=40, height=15, wrap="word")
        self.watched_movies_text.grid(row=1, column=0, padx=10, pady=5)

        # Recommendations Label
        self.results_label = tk.Label(self.display_frame, text="Recommendations", font=("Arial", 12, "bold"))
        self.results_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.recommendations_text = scrolledtext.ScrolledText(self.display_frame, width=40, height=15, wrap="word")
        self.recommendations_text.grid(row=1, column=1, padx=10, pady=5)

        # Movie Links Label
        self.links_label = tk.Label(self.display_frame, text="Movie Ranks (IMDB)", font=("Arial", 12, "bold"))
        self.links_label.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        self.links_text = scrolledtext.ScrolledText(self.display_frame, width=40, height=15, wrap="word")
        self.links_text.grid(row=1, column=2, padx=10, pady=5)

    def get_hybrid_recommendations(self, user_id, top_n, alpha=0.5):
        try:
            user_idx = user_encoder.transform([user_id])[0]
            rated_movies = ratings[ratings['userId'] == user_id]['movieId']
            unrated_movies = movies[~movies['movieId'].isin(rated_movies)]

            if unrated_movies.empty:
                return [], []

            candidate_movies = unrated_movies.copy()
            candidate_movies['userId'] = user_idx

            feature_columns = ['userId'] + [col for col in movies.columns if col not in ['movieId', 'title']]
            X_candidate = torch.tensor(scaler.transform(candidate_movies[feature_columns]), dtype=torch.float32)

            content_based_model.eval()
            with torch.no_grad():
                candidate_movies['content_score'] = content_based_model(X_candidate).numpy().flatten()

            candidate_movies['svd_score'] = candidate_movies['movieId'].apply(lambda x: svd_model.predict(user_idx, x).est)
            candidate_movies['hybrid_score'] = alpha * candidate_movies['content_score'] + (1 - alpha) * candidate_movies['svd_score']

            top_recommendations = candidate_movies.sort_values(by='hybrid_score', ascending=False).head(top_n)
            recommended_titles = list(top_recommendations['title'])
            recommended_links = [self.get_movies_ranks(mid) for mid in top_recommendations['movieId']]

            return recommended_titles, recommended_links
        except Exception as e:
            print(f"Error in hybrid recommendation: {e}")
            return [], []

    def get_movie_link(self, movie_id):
        try:
            imdbId = links[links['movieId'] == movie_id]['imdbId'].values[0]
            return f"https://www.imdb.com/title/tt{imdbId}/"
        except:
            return "No link available"
    
    def get_movies_ranks(self,movie_id):
        #print("index_id",index_id)
        imdbId = links['imdbId'][links['movieId']==movie_id].values[0]
        #print("imdbId",imdbId)

        #url = "https://www.imdb.com/title/tt0114709/"
        # Define headers to make the request look like it's coming from a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
        
        response = requests.get(f"https://www.omdbapi.com/?i=tt{str(imdbId).zfill(7)}&apikey=56280332", headers=headers)
        
        if response.status_code == 200:
            #print("Request successful!")
            data = response.json()
            #print(data)  # This will print the HTML content of the page
            #print(data)
            return data.get('imdbRating', "Rank not found")
            #return str(data)
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return f"Failed to retrieve the page. Status code: {response.status_code}"
    def get_rated_movies(self, user_id):
        watched_movies = ratings[ratings['userId'] == user_id].merge(movies, on='movieId')
        return list(watched_movies['title'])

    def get_recommendations(self):
        try:
            user_id = int(self.user_id_entry.get().strip())
            num_recommendations = int(self.recommendations_entry.get().strip())

            watched_movies = self.get_rated_movies(user_id)
            if not watched_movies:
                watched_movies = ["No watched movies found."]

            recommendations, movie_links = self.get_hybrid_recommendations(user_id, num_recommendations)
            if not recommendations:
                recommendations = ["No recommendations available."]
                movie_links = ["No links available."]

            self.watched_movies_text.delete(1.0, tk.END)
            self.watched_movies_text.insert(tk.END, "\n".join(watched_movies))

            self.recommendations_text.delete(1.0, tk.END)
            self.recommendations_text.insert(tk.END, "\n".join(recommendations))

            self.links_text.delete(1.0, tk.END)
            self.links_text.insert(tk.END, "\n".join(movie_links))

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid integers for User ID and Number of Recommendations.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Initialize the GUI
root = tk.Tk()
app = RecommendationApp(root)
root.mainloop()
