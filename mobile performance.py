import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_excel('mobi.xlsx')

# Select the relevant columns for prediction
selected_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 'Processor',
                    'RAM (MB)', 'Internal storage (GB)', 'Rear camera', 'Front camera', 'Operating system',
                    'Wi-Fi', 'Bluetooth', 'GPS', 'Number of SIMs', '3G', '4G/ LTE', 'Performance']
data = data[selected_columns]

# Preprocessing
data = data.dropna()  # Remove rows with missing values

# Separate the features and target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels
print("Features")

print("Labels")


# Encode categorical features
label_encoder = LabelEncoder()
X['Operating system'] = label_encoder.fit_transform(X['Operating system'])
X['Wi-Fi'] = label_encoder.fit_transform(X['Wi-Fi'])
X['Bluetooth'] = label_encoder.fit_transform(X['Bluetooth'])
X['GPS'] = label_encoder.fit_transform(X['GPS'])
X['3G'] = label_encoder.fit_transform(X['3G'])
X['4G/ LTE'] = label_encoder.fit_transform(X['4G/ LTE'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)


# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
