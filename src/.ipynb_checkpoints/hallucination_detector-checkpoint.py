from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

vectorizer = TfidfVectorizer(
    max_features=10000
)

X = []
y = []

for item in responses:

    text = item["response"]

    label = (
        1 if len(item["labels"]) > 0
        else 0
    )

    X.append(text)
    y.append(label)

X = vectorizer.fit_transform(X)

clf = RandomForestClassifier(
    n_estimators=200
)

clf.fit(X, y)