import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('dataset.csv')
print('\n')
print('Data successfully loaded')
print('Shape: ', df.shape)
print(df.head())

# Visualizing the data
df.plot(x='Investment', y='ROI', style='o')
plt.title('Investment x ROI')
plt.xlabel('Investment')
plt.ylabel('ROI')
plt.savefig('part1-graph1.png')
plt.show()


# Preparing the data
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# Split data into training data and testing data (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Adjust shape and type for training data
X_train = X_train.reshape(-1, 1).astype(np.float32)

# Building the linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)
print('\n')
print('B1 (coef_): ', model.coef_)
print('B0 (intercept_): ', model.intercept_)

# Plot
# y = B0 + B1 * X
regression_line = model.coef_ * X + model.intercept_
plt.scatter(X, y)
plt.title('Investment x ROI')
plt.xlabel('Investment')
plt.ylabel('Predicted ROI')
plt.plot(X, regression_line, color='red')
plt.savefig('part1-regression_line.png')
plt.show()


# Prediction with testing data
y_pred = model.predict(X_test)

# Real x Predicted
df_values = pd.DataFrame({'Real Value': y_test, 'Predicted value': y_pred})
print('\n')
print(df_values)


# Plot
fig, ax = plt.subplots()
idx = np.arange(len(X_test))
bar_width = 0.35
actual = plt.bar(idx, df_values['Real Value'], bar_width, label = 'Real Value')
predicted = plt.bar(idx + bar_width, df_values['Predicted value'], bar_width, label = 'Predicted Value')
plt.xlabel('Investment')
plt.ylabel('Predicted ROI')
plt.title('Real value X Predicted Value')
plt.xticks(idx + bar_width, X_test)
plt.legend()
plt.savefig('part1-actual-vs-predicted.png')
plt.show()


# Evaluation of the model
print('\n')
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', math.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score: ', r2_score(y_test, y_pred))


# Predicting ROI for new data
# Receives terminal input, then apply to the data the same treatment of the training data
print('\n')
input_inv = input('\nType the amount to be invested: ')
input_inv = float(input_inv)
inv = np.array([input_inv])
inv = inv.reshape(-1, 1)

# Predictions
pred_score = model.predict(inv)
print('\n')
print('Amount invested = ', input_inv)
print(' Expected ROI = {:.4}'.format(pred_score[0]))
print('\n')