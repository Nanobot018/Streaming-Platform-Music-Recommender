import pylast
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict, Counter
import logging
import pandas as pd
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO)

# LastFM API configuration
API_KEY = "your_api_key_here"
API_SECRET = "your_secret_key_here"

class MusicRecommender:
    def __init__(self, api_key, api_secret):
        self.network = pylast.LastFMNetwork(api_key=api_key, api_secret=api_secret)
        self.user_dict = {}
        self.artist_dict = {}
        self.model = SVD()
        self.recommendation_count = Counter()
        self.recent_recommendations = []

    def fetch_data(self, users, limit=200):
        user_artist_data = []
        for username in users:
            try:
                user = self.network.get_user(username)
                top_artists = user.get_top_artists(limit=limit)
                for artist in top_artists:
                    artist_name = artist.item.name
                    playcount = int(artist.weight)
                    if username not in self.user_dict:
                        self.user_dict[username] = len(self.user_dict)
                    if artist_name not in self.artist_dict:
                        self.artist_dict[artist_name] = len(self.artist_dict)
                    user_artist_data.append((username, artist_name, playcount))
                logging.info(f"Fetched {len(top_artists)} artists for user {username}")
            except pylast.WSError as e:
                logging.error(f"Error fetching data for user {username}: {str(e)}")
            except Exception as e:
                logging.error(f"Unexpected error for user {username}: {str(e)}")
        
        logging.info(f"Total data points fetched: {len(user_artist_data)}")
        return user_artist_data

    def train_model(self, user_artist_data):
        if not user_artist_data:
            logging.error("No data available to train the model.")
            return

        df = pd.DataFrame(user_artist_data, columns=['user', 'artist', 'playcount'])
        df['normalized_playcount'] = 1 + (df['playcount'] - df['playcount'].min()) / (df['playcount'].max() - df['playcount'].min()) * 4
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user', 'artist', 'normalized_playcount']], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        logging.info("Model training completed.")

    def get_recommendations(self, username, n=10):
        if username not in self.user_dict:
            logging.warning(f"User {username} not found in the dataset.")
            return None, 0
        
        user_items = set(artist for artist, _ in self.model.trainset.ur[self.user_dict[username]])
        all_items = set(self.artist_dict.keys())
        candidates = list(all_items - user_items)
        
        predictions = [self.model.predict(username, artist) for artist in candidates]
        top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n*2]  # Get more candidates
        
        # Filter out recently recommended artists
        filtered_top_n = [pred for pred in top_n if pred.iid not in self.recent_recommendations]
        
        if not filtered_top_n:
            filtered_top_n = top_n  # If all are recent, use the original list
        
        recommended_artist = random.choice(filtered_top_n).iid
        self.recommendation_count[recommended_artist] += 1
        
        # Update recent recommendations
        self.recent_recommendations.append(recommended_artist)
        if len(self.recent_recommendations) > n:
            self.recent_recommendations.pop(0)
        
        return recommended_artist, self.recommendation_count[recommended_artist]

def main():
    print("Welcome to Your Personal Music Recommender!")
    print("-------------------------------------------")
    
    username = input("Please enter your Last.fm username: ")
    
    print(f"\nFetching your music data, {username}. This may take a moment...")
    recommender = MusicRecommender(API_KEY, API_SECRET)
    user_artist_data = recommender.fetch_data([username])
    
    if not user_artist_data:
        print("Sorry, we couldn't fetch any data for your account.")
        return
    
    print("Data fetched successfully!")
    print("Training the recommendation model...")
    recommender.train_model(user_artist_data)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Get a music recommendation")
        print("2. Exit")
        
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            print("\nGenerating a recommendation for you...")
            time.sleep(1)  # Add a small delay for effect
            artist, count = recommender.get_recommendations(username, n=10)
            
            if artist:
                print(f"\nBased on your listening history, you might enjoy:")
                print(f"Artist: {artist}")
                print(f"Times recommended: {count}")
            else:
                print("Sorry, we couldn't generate a recommendation at this time.")
        
        elif choice == '2':
            print("Thank you for using Your Personal Music Recommender. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
