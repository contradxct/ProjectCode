import requests
import time
import math
import json
import csv
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score

client_id = "389f206f9006cdcdea191693b4f9a0e1"

url = "https://api.myanimelist.net/v2/anime/ranking"
headers = {"X-MAL-CLIENT-ID": client_id}

params_base = {
    "ranking_type": "all",
    "limit": 100,  
    "fields": "id,title,mean,num_episodes,rating,popularity,genres"
}


all_data = {}   

# DATA COLLECTION SECTION

for offset in range(0, 2000, 100):
    params = params_base.copy()
    params["offset"] = offset

    response = requests.get(url, headers=headers, params=params)
    batch = response.json().get("data", [])

    for item in batch:
        anime = item["node"]
        all_data[anime["id"]] = anime
    time.sleep(1)   

print(f"Total unique anime collected: {len(all_data)}")

# CSV FILE EXPORT SECTION

csv_file = "anime_sample.csv"

with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Title", "Mean", "Episodes", "Rating", "Popularity", "Genres"])

    for anime in all_data.values():
        genres = ", ".join([g["name"] for g in anime.get("genres", [])])

        writer.writerow([
            anime["id"],
            anime["title"],
            anime.get("mean"),
            anime.get("num_episodes"),
            anime.get("rating"),
            anime.get("popularity"),
            genres
        ])

print(f"Data saved to {csv_file}!")

# SQL DATABASE STORAGE SECTION

conn = sqlite3.connect("anime.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS anime (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    mean FLOAT,
    episodes INTEGER,
    rating TEXT,
    popularity INTEGER
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS anime_genres (
    anime_id INTEGER NOT NULL,
    genre TEXT NOT NULL,
    PRIMARY KEY (anime_id, genre),
    FOREIGN KEY (anime_id) REFERENCES anime(id)
);
""")

for anime in all_data.values():
    cursor.execute("""
        INSERT OR IGNORE INTO anime (id, title, mean, episodes, rating, popularity)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        anime["id"],
        anime["title"],
        anime.get("mean"),
        anime.get("num_episodes"),
        anime.get("rating"),
        anime.get("popularity")
    ))

    for g in anime.get("genres", []):
        cursor.execute("""
            INSERT OR IGNORE INTO anime_genres (anime_id, genre)
            VALUES (?, ?)
        """, (
            anime["id"],
            g["name"]
        ))

conn.commit()

# READING BACK SQL DATA

df = pd.read_sql("""
    SELECT a.id, a.title, a.mean, a.episodes, a.rating, a.popularity,
           GROUP_CONCAT(ag.genre) AS genres
    FROM anime a
    LEFT JOIN anime_genres ag ON a.id = ag.anime_id
    GROUP BY a.id
""", conn)

conn.close()

print(df.isnull().sum())

# BASIC DATA CLEANING

df = df.drop_duplicates()
df = df[df["episodes"].notnull()]

#Exploratory Data Analysis (EDA) SECTION

plt.figure(figsize=(8,5))
plt.hist(df["mean"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Anime Ratings (Mean)")
plt.xlabel("Mean Rating")
plt.ylabel("Number of Anime")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["episodes"], bins=30, color="lightgreen", edgecolor="black")
plt.title("Distribution of Number of Episodes")
plt.xlabel("Episodes")
plt.ylabel("Number of Anime")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["popularity"], bins=20, color="salmon", edgecolor="black")
plt.title("Distribution of Anime Popularity (Ranking)")
plt.xlabel("Popularity")
plt.ylabel("Number of Anime")
plt.show()

print(df[["mean", "episodes", "popularity"]].describe())
print(df[["mean", "episodes", "popularity"]].corr())

# GENRE ANALYSIS

all_genres = []
for g_list in df["genres"].dropna():
    genres = [g.strip() for g in g_list.split(",")]
    all_genres.extend(genres)
genre_counts = Counter(all_genres)

print(genre_counts.most_common(10))

top_genres = genre_counts.most_common(10)
labels, counts = zip(*top_genres)

plt.figure(figsize=(10,5))
plt.bar(labels, counts, color="purple")
plt.title("Top 10 Anime Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Anime")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot(df["episodes"])
plt.title("Boxplot of Number of Episodes")
plt.ylabel("Episodes")
plt.show()

# Feature Engineering + ML SECTION

df["popular_label"] = (df["popularity"] <= 2000).astype(int)
df["episodes_log"] = df["episodes"].apply(lambda x: 0 if x == 0 else math.log(x))
X = df[["mean", "episodes", "episodes_log"]].fillna(0)
y = df["popular_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

#Additional Evaluation Metrics

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("ROC-AUC:", roc)

#Sample Predictions

sample_df = df.sample(10)[["title", "mean", "episodes", "episodes_log", "popularity", "popular_label"]].fillna(0)
X_sample = sample_df[["mean", "episodes", "episodes_log"]]
predictions = model.predict(X_sample)

for i, (index, row) in enumerate(sample_df.iterrows()):
    predicted = predictions[i]
    print(f"Sample {i+1}: {row['title']}, Predicted Popular: {predicted}, Actual Ranking: {row['popularity']}")