# Step 1: Import libraries
import numpy as np  # For numerical computations and arrays
import pandas as pd  # For handling tabular data
import matplotlib.pyplot as plt  # For plotting visualizations
from sklearn.datasets import fetch_california_housing  # Load dataset from Scikit-learn
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.preprocessing import StandardScaler  # Normalize data
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics
import torch  # PyTorch for deep learning model
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizer for PyTorch
from tensorflow import keras  # Keras/TensorFlow for deep learning
import tensorflow as tf  # TensorFlow core

# Step 2: Load and explore data (Pandas and NumPy)
housing = fetch_california_housing(as_frame=True)  # Load data as a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)  # Convert to Pandas DataFrame
df['Price'] = housing.target  # Add target column (house prices in million USD)
print(df.head())  # Display first 5 rows
print(df.describe())  # Show descriptive statistics (mean, std, etc. using NumPy)

# Step 3: Visualize data (Matplotlib and NumPy)
plt.figure(figsize=(10, 4))  # Set figure size
plt.subplot(1, 2, 1)  # First subplot
plt.hist(df['Price'], bins=50, color='blue')  # Histogram of prices (uses NumPy arrays)
plt.title('Distribution of House Prices')
plt.xlabel('Price (Million USD)')

plt.subplot(1, 2, 2)  # Second subplot
x = df['MedInc'].values  # Median income as NumPy array
y = df['Price'].values  # Prices as NumPy array
plt.scatter(x, y, alpha=0.5)  # Scatter plot
plt.title('Median Income vs Price')
plt.xlabel('Median Income')
plt.ylabel('Price')

plt.tight_layout()  # Adjust layout
plt.show()  # Display plots (shows positive correlation between income and price)

# Step 4: Preprocess data (NumPy, Pandas, Scikit-learn)
X = df.drop('Price', axis=1).values  # Features to NumPy array
y = df['Price'].values  # Target to NumPy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test
scaler = StandardScaler()  # Initialize scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on train data
X_test_scaled = scaler.transform(X_test)  # Transform test data
print(f"Shape of training data: {X_train_scaled.shape}")  # E.g., (16512, 8)

# Step 5: Scikit-learn model (Linear Regression)
model_sk = LinearRegression()  # Linear model: y = Xβ + ε
model_sk.fit(X_train_scaled, y_train)  # Train model
y_pred_sk = model_sk.predict(X_test_scaled)  # Predict on test data
mse = mean_squared_error(y_test, y_pred_sk)  # Mean squared error
r2 = r2_score(y_test, y_pred_sk)  # R-squared score
print(f"Scikit-learn MSE: {mse:.2f}, R²: {r2:.2f}")  # E.g., MSE: 0.53, R²: 0.58

# Step 6: PyTorch model
# Convert data to PyTorch tensors (from NumPy)
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)  # Features to tensor
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Target to tensor (shape: n,1)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)  # Test features
y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Test target

# Define neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # Input: 8 features, output: 64 neurons
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output: price
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = torch.relu(self.fc2(x))  # ReLU activation
        return self.fc3(x)  # Linear output

model_pt = Net()  # Initialize model
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)  # Adam optimizer

# Training loop
for epoch in range(10):  # 10 epochs for simplicity
    optimizer.zero_grad()  # Clear gradients
    outputs = model_pt(X_train_torch)  # Forward pass
    loss = criterion(outputs, y_train_torch)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model_pt.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    y_pred_pt = model_pt(X_test_torch).numpy()  # Convert predictions to NumPy
mse_pt = mean_squared_error(y_test, y_pred_pt)  # Compute MSE
r2_pt = r2_score(y_test, y_pred_pt)  # Compute R²
print(f"PyTorch MSE: {mse_pt:.2f}, R²: {r2_pt:.2f}")

# Step 7: Keras/TensorFlow model
model_tf = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(8,)),  # First layer
    keras.layers.Dense(32, activation='relu'),  # Second layer
    keras.layers.Dense(1)  # Output layer
])
model_tf.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Compile with Adam and MSE
history = model_tf.fit(X_train_scaled, y_train, epochs=10, batch_size=32, 
                       validation_split=0.2, verbose=1)  # Train with 20% validation
y_pred_tf = model_tf.predict(X_test_scaled).flatten()  # Predict and flatten output
mse_tf = mean_squared_error(y_test, y_pred_tf)  # Compute MSE
r2_tf = r2_score(y_test, y_pred_tf)  # Compute R²
print(f"TensorFlow MSE: {mse_tf:.2f}, R²: {r2_tf:.2f}")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('TensorFlow Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 8: Compare results
results = pd.DataFrame({
    'Model': ['Scikit-learn', 'PyTorch', 'TensorFlow'],
    'MSE': [mse, mse_pt, mse_tf],
    'R²': [r2, r2_pt, r2_tf]
})
print(results)  # Display comparison table
# Conclusion: Deep learning models (PyTorch/TF) usually have better R² with more training
