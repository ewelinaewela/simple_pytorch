import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# Load trained model
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load model
model = SimpleNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Load and scale dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🎯 Add sidebar for user inputs
st.sidebar.header("🔢 Adjust Input Features")
x1 = st.sidebar.slider("Feature 1 (x1)", float(X[:, 0].min()), float(X[:, 0].max()), 0.0)
x2 = st.sidebar.slider("Feature 2 (x2)", float(X[:, 1].min()), float(X[:, 1].max()), 0.0)

# Function to classify user input
def predict_class(x1, x2):
    input_data = np.array([[x1, x2]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

# 🎯 Add "Classify" button
if st.sidebar.button("🔍 Classify Point"):
    pred = predict_class(x1, x2)
    class_label = "🔵 Class 0" if pred < 0.5 else "🔴 Class 1"
    st.sidebar.write(f"**Prediction:** {class_label} ({pred:.2f})")

# Function to plot decision boundary
def plot_decision_boundary():
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid_scaled = scaler.transform(X_grid)
    X_grid_tensor = torch.FloatTensor(X_grid_scaled)
    
    with torch.no_grad():
        Z = model(X_grid_tensor).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    plt.xlabel("Feature 1 (x1)")
    plt.ylabel("Feature 2 (x2)")
    plt.title("Decision Boundary of Neural Network")
    st.pyplot(plt)

# Display decision boundary
plot_decision_boundary()

