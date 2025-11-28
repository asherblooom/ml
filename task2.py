
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)
X = data.drop(columns=['charges'])
y = data['charges'].values.reshape(-1, 1)

# one hot encode the categorical columns, scale the numerical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_cols), 
                  ('num', StandardScaler(), numerical_cols)],
    verbose_feature_names_out=False
)
y_scaler = StandardScaler()

# Split into training and testing sets (80% / 20%)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train_raw.shape)
# fit preprocessor on training data, then transform both
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
y_train = y_scaler.fit_transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 4. Define Neural Network Architecture
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
class NN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 32) # Hidden Layer 1
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)        # Hidden Layer 2
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)         # Output Layer

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

# Initialise Model
model = NN(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 500
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_preds_scaled = model(X_test_tensor).numpy()
    train_preds_scaled = model(X_train_tensor).numpy()

# Inverse transform to get actual dollar values
test_preds = y_scaler.inverse_transform(test_preds_scaled)
train_preds = y_scaler.inverse_transform(train_preds_scaled)
y_test_orig = y_scaler.inverse_transform(y_test)
y_train_orig = y_scaler.inverse_transform(y_train)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_preds))

print("-" * 30)
print(f"Training RMSE: ${train_rmse:.2f}")
print(f"Test RMSE:     ${test_rmse:.2f}")

# Plot Results
plt.figure()
plt.scatter(y_test_orig, test_preds, alpha=0.5)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()])
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('PyTorch NN: Predicted vs Actual')
plt.show()