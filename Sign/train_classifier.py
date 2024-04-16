import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data from pickle file
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

X = data['data']
y = data['labels']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Train SVM classifier
svm.fit(X_train, y_train)

# Save the trained SVM model
with open('svm_model.pickle', 'wb') as f:
    pickle.dump(svm, f)

# Predict labels for test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
