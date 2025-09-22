# Housing-price-
#you should download data base from this link and call it in program which I was wrote it.
# گام ۱: Import کتابخانه‌ها
import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score  
import torch  
import torch.nn as nn
import torch.optim as optim
from tensorflow import keras  
import tensorflow as tf

# گام ۲: بارگیری و کاوش داده‌ها (Pandas و NumPy)
housing = fetch_california_housing(as_frame=True) 
df = pd.DataFrame(housing.data, columns=housing.feature_names)  
df['Price'] = housing.target  
print(df.head())  
print(df.describe())  

# گام ۳
plt.figure(figsize=(10, 4))  
plt.subplot(1, 2, 1)  
plt.hist(df['Price'], bins=50, color='blue')  
plt.title('توزیع قیمت خانه‌ها')
plt.xlabel('قیمت (میلیون دلار)')

plt.subplot(1, 2, 2)  
x = df['MedInc'].values  
y = df['Price'].values  
plt.scatter(x, y, alpha=0.5)  
plt.title('درآمد متوسط vs قیمت')
plt.xlabel('درآمد متوسط')
plt.ylabel('قیمت')

plt.tight_layout()
plt.show()  

# گام ۴: پیش‌پردازش داده‌ها (NumPy، Pandas، Scikit-learn)
X = df.drop('Price', axis=1).values  
y = df['Price'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  
print(f"شکل داده‌های train: {X_train_scaled.shape}")  

# گام ۵: مدل Scikit-learn (رگرسیون خطی)
model_sk = LinearRegression()  
model_sk.fit(X_train_scaled, y_train)  
y_pred_sk = model_sk.predict(X_test_scaled)  
mse = mean_squared_error(y_test, y_pred_sk)  
r2 = r2_score(y_test, y_pred_sk)  # ضریب تعیین
print(f"Scikit-learn MSE: {mse:.2f}, R²: {r2:MSE: 0.53, R²: 0.58

# گام ۶: مدل PyTorch
# تبدیل داده‌ها به تنسور (از NumPy)
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # شکل (n,1)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# تعریف مدل
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # ۸ ورودی به ۶۴ نورون
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # خروجی: قیمت
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # فعال‌سازی ReLU
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model_pt = Net()
criterion = nn.MSELoss()  # تابع ضرر
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)  # بهینه‌ساز

# آموزش
for epoch in range(10):  # ۱۰ epoch برای سادگی
    optimizer.zero_grad()  # صفر کردن گرادیان
    outputs = model_pt(X_train_torch)  # forward
    loss = criterion(outputs, y_train_torch)  # محاسبه ضرر
    loss.backward()  # backpropagation
    optimizer.step()  # به‌روزرسانی وزن‌ها
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ارزیابی
model_pt.eval()
with torch.no_grad():
    y_pred_pt = model_pt(X_test_torch).numpy()  # به NumPy برای ارزیابی
mse_pt = mean_squared_error(y_test, y_pred_pt)
r2_pt = r2_score(y_test, y_pred_pt)
print(f"PyTorch MSE: {mse_pt:.2f}, R²: {r2_pt:.2f}")

# گام ۷: مدل Keras/TensorFlow
model_tf = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(8,)),  # لایه اول
    keras.layers.Dense(32, activation='relu'),  # لایه دوم
    keras.layers.Dense(1)  # خروجی
])
model_tf.compile(optimizer='adam', loss='mse', metrics=['mae'])  # کامپایل
history = model_tf.fit(X_train_scaled, y_train, epochs=10, batch_size=32, 
                       validation_split=0.2, verbose=1)  # آموزش با ۲۰% validation
y_pred_tf = model_tf.predict(X_test_scaled).flatten()  # پیش‌بینی
mse_tf = mean_squared_error(y_test, y_pred_tf)
r2_tf = r2_score(y_test, y_pred_tf)
print(f"TensorFlow MSE: {mse_tf:.2f}, R²: {r2_tf:.2f}")

# رسم تاریخچه ضرر
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('تاریخچه آموزش TensorFlow')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# گام ۸: مقایسه نتایج
results = pd.DataFrame({
    'Model': ['Scikit-learn', 'PyTorch', 'TensorFlow'],
    'MSE': [mse, mse_pt, mse_tf],
    'R²': [r2, r2_pt, r2_tf]
})
print(results) 
