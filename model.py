import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("rent_data.csv")

# Input and output
X = data[['area_sqft', 'bedrooms', 'location']]
y = data['rent']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("rent_model.pkl", "wb"))

print("Model trained successfully!")
