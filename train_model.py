# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras_tuner import BayesianOptimization
import matplotlib.pyplot as plt
import joblib  # Import joblib for saving models

# Load the data
file_path = 'data/data.csv'
data = pd.read_csv(file_path)

# Print the columns of the DataFrame to check what's available
print("Available columns:", data.columns.tolist())

# Remove non-numeric and unnecessary columns
data = data.select_dtypes(include=[np.number])

# Handle missing values by filling with mean
data.fillna(data.mean(), inplace=True)

# Check if the columns you want to drop exist
columns_to_drop = ['cursor_x', 'cursor_y',]

# Filter the columns to drop to only those that exist in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
print("Dropping columns:", existing_columns_to_drop)

# Split features and target
X = data.drop(columns=existing_columns_to_drop)  # Drop target columns from features
y = data[['cursor_y']]  # Using only cursor_y as target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# Apply PCA
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Model building function
def build_model(hp):
    model = Sequential([
        Dense(hp.Int('units_1', min_value=256, max_value=512, step=128), activation='relu', input_shape=(X_train_pca.shape[1],)),
        BatchNormalization(),
        Dropout(hp.Choice('dropout_1', values=[0.3, 0.4, 0.5])),
        
        Dense(hp.Int('units_2', min_value=128, max_value=256, step=64), activation='relu'),
        BatchNormalization(),
        Dropout(hp.Choice('dropout_2', values=[0.3, 0.4])),
        
        Dense(hp.Int('units_3', min_value=64, max_value=128, step=32), activation='relu'),
        Dense(1)  # Output layer for cursor_y only
    ])
    
    # Compile with Adam and tunable learning rate
    lr_schedule = ExponentialDecay(initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'), decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse', metrics=['mae'])
    return model

# Use Keras Tuner for Bayesian optimization over hyperparameters
tuner = BayesianOptimization(build_model, objective='val_loss', max_trials=5, executions_per_trial=3, directory='my_dir3', project_name='gaze_model_tuning')

# Search for the best model
tuner.search(X_train_pca, y_train, epochs=100, validation_split=0.15)
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate on test set
y_pred = best_model.predict(X_test_pca)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate the highest error
errors = np.abs(y_test.values - y_pred)  # Calculate absolute errors
max_error = np.max(errors)  # Find the maximum error

print(f"Optimized Mean Squared Error: {mse}")
print(f"Optimized Mean Absolute Error: {mae}")
print(f"Optimized R2 Score: {r2}")
print(f"Highest Error: {max_error}")

# Save the scaler and PCA model
joblib.dump(scaler, 'model/scaler_y_updates_70000.pkl')
joblib.dump(pca, 'model/pca_model_y_updated_70000.pkl')

# Save the trained model
best_model.save("model/gaze_detection_model_y1_updated_70000.h5")

# Scatter plot of actual vs predicted values
plt.figure(figsize=(6, 6))
plt.scatter(y_test['cursor_y'], y_pred, alpha=0.5)
plt.title('Actual vs Predicted cursor_y')
plt.xlabel('Actual cursor_y')
plt.ylabel('Predicted cursor_y')
plt.plot([y_test['cursor_y'].min(), y_test['cursor_y'].max()], [y_test['cursor_y'].min(), y_test['cursor_y'].max()], color='red', linestyle='--')  # Diagonal line
plt.tight_layout()
plt.show()


# # Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.decomposition import PCA
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from keras_tuner import BayesianOptimization
# import matplotlib.pyplot as plt
# import joblib  # Import joblib for saving models

# # Load the data
# file_path = 'data/data.csv'
# data = pd.read_csv(file_path)

# # Print the columns of the DataFrame to check what's available
# print("Available columns:", data.columns.tolist())

# # Remove non-numeric and unnecessary columns
# data = data.select_dtypes(include=[np.number])

# # Handle missing values by filling with mean
# data.fillna(data.mean(), inplace=True)

# # Check if the columns you want to drop exist
# columns_to_drop = ['cursor_x', 'cursor_y',
#                    'head_roll267', 'head_roll269', 'head_roll270', 'head_roll409', 'head_roll375',
#                    'head_roll321', 'head_roll405', 'head_roll314', 'head_roll17', 'head_roll84',
#                    'head_roll181', 'head_roll91', 'head_roll146', 'head_roll37', 'head_roll39',
#                    'head_roll40', 'head_roll185', 'head_roll61','normalized_roll_angle']

# # Filter the columns to drop to only those that exist in the DataFrame
# existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
# print("Dropping columns:", existing_columns_to_drop)

# # Split features and target
# X = data.drop(columns=existing_columns_to_drop)  # Drop target columns from features
# print(X)
# y = data[['cursor_y']]  # Using only cursor_y as target

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Reshape data for LSTM (samples, time_steps, features)
# time_steps = 5  # You can adjust the number of time steps (frames) per sample

# def create_sequences(data, time_steps):
#     sequences = []
#     for i in range(len(data) - time_steps):
#         sequences.append(data[i:i+time_steps])
#     return np.array(sequences)

# # Create sequences for the input data (X)
# X_seq = create_sequences(X_scaled, time_steps)

# # Create corresponding sequences for the target data (y)
# y_seq = y.iloc[time_steps:].values

# # Train-Test split
# X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.15, random_state=42)

# # Apply PCA for dimensionality reduction (if needed)
# pca = PCA(n_components=0.99)
# # X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], 1,X_train.shape[1]))  # Reshape for PCA
# # print(X_train_pca.shape)
# # X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))  # Reshape for PCA
# X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
# X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))
# print(X_train_pca.shape)
# X_train_pca = X_train_pca.reshape((X_train_pca.shape[0], 1, X_train_pca.shape[1]))  # (samples, 1, features)
# X_test_pca = X_test_pca.reshape((X_test_pca.shape[0], 1, X_test_pca.shape[1]))  # (samples, 1, features)

# # Model building function
# def build_model(hp):
#     model = Sequential([
#         LSTM(hp.Int('units_1', min_value=64, max_value=256, step=64), activation='relu', input_shape=(X_train_pca.shape[1], X_train_pca.shape[2]), return_sequences=True),
#         Dropout(hp.Choice('dropout_1', values=[0.3, 0.4, 0.5])),
#         BatchNormalization(),
        
#         LSTM(hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu', return_sequences=False),
#         Dropout(hp.Choice('dropout_2', values=[0.3, 0.4])),
        
#         Dense(1)  # Output layer for cursor_y only
#     ])
    
#     # Compile with Adam and tunable learning rate
#     lr_schedule = ExponentialDecay(initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'), decay_steps=10000, decay_rate=0.9)
#     model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse', metrics=['mae'])
#     return model

# # Use Keras Tuner for Bayesian optimization over hyperparameters
# tuner = BayesianOptimization(build_model, objective='val_loss', max_trials=5, executions_per_trial=3, directory='my_dir3', project_name='gaze_model_tuning')

# # Search for the best model
# tuner.search(X_train_pca, y_train, epochs=100, validation_split=0.15)
# best_model = tuner.get_best_models(num_models=1)[0]

# # Evaluate on test set
# y_pred = best_model.predict(X_test_pca)

# # Calculate performance metrics
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Calculate the highest error
# errors = np.abs(y_test - y_pred)  # Calculate absolute errors
# max_error = np.max(errors)  # Find the maximum error

# print(f"Optimized Mean Squared Error: {mse}")
# print(f"Optimized Mean Absolute Error: {mae}")
# print(f"Optimized R2 Score: {r2}")
# print(f"Highest Error: {max_error}")

# # Save the scaler and PCA model
# joblib.dump(scaler, 'model/scaler_y_lstm.pkl')
# joblib.dump(pca, 'model/pca_model_y_lstm.pkl')

# # Save the trained model
# best_model.save("model/gaze_detection_model_y_lstm.h5")

# # Scatter plot of actual vs predicted values
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.title('Actual vs Predicted cursor_y')
# plt.xlabel('Actual cursor_y')
# plt.ylabel('Predicted cursor_y')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Diagonal line
# plt.tight_layout()
# plt.show()
