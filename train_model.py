import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("products.csv")

# Clean text
df["Product Title"] = (
    df["Product Title"]
    .astype(str)
    .str.lower()
    .str.replace(r"[^a-z0-9 ]", " ", regex=True)
)

df = df.dropna(subset=[" Category Label"])

X = df["Product Title"]
y = df[" Category Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
    ("clf", RandomForestClassifier(n_estimators=300))
])

model.fit(X_train, y_train)

# Save model
with open("product_category_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as product_category_model.pkl")