import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('data/data.csv')  # Replace with your actual dataset path

# Extract only the cursor_x and cursor_y columns
x = data['cursor_x']
y = data['cursor_y']

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, c='blue', edgecolors='k', s=20)
plt.xlabel('Cursor X')
plt.ylabel('Cursor Y')
plt.title('Spread of Cursor Points')
plt.grid(True)
plt.show()
