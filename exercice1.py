import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer1(x)
        return x

def train_model(epochs, lr):
    # Data for [-1, 1]
    x_vals = np.linspace(-1, 1, 20000)
    y_vals = np.sin(x_vals)
    x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)

    # Initialize model
    model = MLP()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(y_tensor)
        loss = criterion(output, x_tensor)
        loss.backward()
        optimizer.step()

    return model

st.title("MLP Training with PyTorch on sin function")
st.sidebar.header("Parameters")

# Sidebar parameters
epochs = st.sidebar.slider("Epochs", 100, 10000, 1000, step=100)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)

if st.button("Launch demo"):
    st.write("### Model Training")
    st.write("Training the model with the specified parameters...")

    model = train_model(epochs, learning_rate)
    
    # Evaluate Model
    model.eval()
    with torch.no_grad():
        # Data for [-1, 1]
        x_vals = np.linspace(-1, 1, 20000)
        y_vals = np.sin(x_vals)
        x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
        y_tensor = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)

        predicted_x = model(y_tensor).numpy().flatten()
        true_x = np.arcsin(y_vals)

        # Mean Squared Error
        mse = np.mean((predicted_x - true_x) ** 2)
        st.write(f"Mean Squared Error (MSE) for y in [-1, 1]: {mse}")

        # Plot Results for [-1, 1]
        fig, ax = plt.subplots(figsize=(8, 4)) 
        ax.plot(y_vals, true_x, label='True arcsin(y)', color='red')
        ax.plot(y_vals, predicted_x, label='Predicted x (MLP)', color='blue', linestyle='--')
        ax.set_xlabel('y = sin(x)')
        ax.set_ylabel('x')
        ax.set_title('Predicted vs True arcsin(y) for y in [-1, 1]')
        ax.legend()
        ax.grid(True)
        ax.text(0.5, 0.1, f'MSE = {mse:.5f}', fontsize=12, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        st.pyplot(fig)

        # Data for [-2, 2]
        x_vals_large = np.linspace(-2, 2, 40000)
        y_vals_large = np.sin(x_vals_large)
        x_tensor_large = torch.tensor(x_vals_large, dtype=torch.float32).view(-1, 1)
        y_tensor_large = torch.tensor(y_vals_large, dtype=torch.float32).view(-1, 1)

        predicted_x_large = model(y_tensor_large).numpy().flatten()
        true_x_large = np.arcsin(y_vals_large)

        # Mean Squared Error for [-2, 2]
        mse_large = np.mean((predicted_x_large - true_x_large) ** 2)
        st.write(f"Mean Squared Error (MSE) for y in [-2, 2]: {mse_large}")

        # Plot Results for [-2, 2]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_vals_large, true_x_large, label='True arcsin(y)', color='red')
        ax.plot(y_vals_large, predicted_x_large, label='Predicted x (MLP)', color='blue', linestyle='--')
        ax.set_xlabel('y = sin(x)')
        ax.set_ylabel('x')
        ax.set_title('Predicted vs True arcsin(y) for y in [-2, 2]')
        ax.legend()
        ax.grid(True)
        ax.text(0.5, 0.1, f'MSE = {mse_large:.5f}', fontsize=12, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))

        st.pyplot(fig)
