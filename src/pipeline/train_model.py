import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, SVDpp, NMF
from surprise.model_selection import train_test_split
from models.recommender import RecommenderSystem
from src.config import RATINGS_PATH, BEST_MODEL_PATH

# Load data
ratings = pd.read_csv(RATINGS_PATH)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Define models
models = [
    RecommenderSystem(SVD(), "SVD"),
    RecommenderSystem(KNNBasic(), "KNNBasic"),
    RecommenderSystem(SVDpp(), "SVDpp"),
    RecommenderSystem(NMF(), "NMF")
]

# Evaluate and select best
results = []
best_score = float('inf')
best_model = None

for model in models:
    model.train(trainset)
    scores = model.evaluate(testset)
    results.append(scores)
    if scores["rmse"] < best_score:
        best_score = scores["rmse"]
        best_model = model

# Show results
for r in results:
    print(f"{r['model']} - RMSE: {r['rmse']:.4f}, MAE: {r['mae']:.4f}")

# Save best model
best_model.save(BEST_MODEL_PATH)
print(f"\n✅ Best model saved: {best_model.name} at {BEST_MODEL_PATH}")

