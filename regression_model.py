import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CREATE SAMPLE DATASET

data = {
    "Area": [600, 700, 800, 900, 1000,
             1100, 1200, 1300, 1400, 1500,
             1600, 1700, 1800, 1900, 2000,
             2100, 2200, 2300, 2400, 2500],

    "Price": [38, 43, 47, 50, 56,
              60, 66, 69, 75, 78,
              83, 89, 92, 97, 103,
              107, 113, 116, 122, 127]
}

df = pd.DataFrame(data)

print("HOUSE PRICE PREDICTION")

print("\nDataset:")
print(df)

# 2. SELECT INPUT AND OUTPUT

X = df[["Area"]].values
y = df["Price"].values

# 3. SPLIT DATA INTO TRAINING AND TESTING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# 4. TAKE DYNAMIC PARAMETERS FROM USER

print("\n------------------------------------------")
print("Gradient Descent Settings")
print("--------------------------------------------")

learning_rate = float(
    input("Enter learning rate (example 0.01): ")
)

iterations = int(
    input("Enter number of iterations (example 1000): ")
)

# 5. STANDARDIZE INPUT DATA

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. GRADIENT DESCENT FUNCTION

def gradient_descent(X, y, learning_rate, iterations):

    m = 0
    c = np.mean(y)

    cost_history = []

    for i in range(iterations):

        # Calculate prediction
        y_pred = m * X.ravel() + c

        # Calculate error
        error = y_pred - y

        # Calculate cost
        cost = np.mean(error ** 2)

        cost_history.append(cost)

        # Calculate gradients
        dm = (2 / len(X)) * np.sum(
            X.ravel() * error
        )

        dc = (2 / len(X)) * np.sum(error)

        # Update parameters
        m = m - learning_rate * dm
        c = c - learning_rate * dc

    return m, c, cost_history

# 7. TRAIN MODEL USING GRADIENT DESCENT

m, c, cost_history = gradient_descent(
    X_train_scaled,
    y_train,
    learning_rate,
    iterations
)

# 8. MAKE PREDICTIONS

gd_predictions = (
    m * X_test_scaled.ravel() + c
)

# 9. EVALUATE GRADIENT DESCENT MODEL

gd_mae = mean_absolute_error(
    y_test,
    gd_predictions
)

gd_mse = mean_squared_error(
    y_test,
    gd_predictions
)

gd_rmse = np.sqrt(gd_mse)

gd_r2 = r2_score(
    y_test,
    gd_predictions
)

# 10. STANDARD LINEAR REGRESSION FOR COMPARISON

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

lr_predictions = linear_model.predict(
    X_test
)

# 11. EVALUATE STANDARD LINEAR REGRESSION

lr_mae = mean_absolute_error(
    y_test,
    lr_predictions
)

lr_mse = mean_squared_error(
    y_test,
    lr_predictions
)

lr_rmse = np.sqrt(lr_mse)

lr_r2 = r2_score(
    y_test,
    lr_predictions
)

# 12. DISPLAY MODEL COMPARISON

print("\n-----------------------------------------")
print("        MODEL PERFORMANCE")
print("-------------------------------------------")

print("\nGradient Descent Linear Regression")
print("------------------------------------------")
print("MAE  :", round(gd_mae, 4))
print("MSE  :", round(gd_mse, 4))
print("RMSE :", round(gd_rmse, 4))
print("R2   :", round(gd_r2, 4))

print("\nScikit-learn Linear Regression")
print("------------------------------------------")
print("MAE  :", round(lr_mae, 4))
print("MSE  :", round(lr_mse, 4))
print("RMSE :", round(lr_rmse, 4))
print("R2   :", round(lr_r2, 4))

# 13. DISPLAY ACTUAL VS PREDICTED VALUES

result = pd.DataFrame({
    "Area": X_test.ravel(),
    "Actual Price": y_test,
    "GD Predicted": np.round(gd_predictions, 2),
    "Linear Regression": np.round(lr_predictions, 2)
})

print("\n-------------------------------------------")
print("       TEST DATA PREDICTIONS")
print("---------------------------------------------")

print(result)

# 14. DYNAMIC HOUSE PRICE PREDICTION

print("\n------------------------------------------")
print("       NEW HOUSE PRICE PREDICTION")
print("--------------------------------------------")

new_area = float(
    input("\nEnter house area in sq.ft.: ")
)

new_area_array = np.array([[new_area]])

new_area_scaled = scaler.transform(
    new_area_array
)

predicted_price = (
    m * new_area_scaled.ravel()[0] + c
)

print(
    "\nEstimated House Price:",
    round(predicted_price, 2),
    "lakh"
)

# 15. GRAPH 1 - REGRESSION LINE

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

x_line = np.linspace(
    X.min(),
    X.max(),
    100
).reshape(-1, 1)

x_line_scaled = scaler.transform(
    x_line
)

y_line = (
    m * x_line_scaled.ravel() + c
)

plt.plot(
    x_line,
    y_line,
    label="Gradient Descent Regression Line"
)

plt.scatter(
    new_area,
    predicted_price,
    marker="*",
    s=150,
    label="New Prediction"
)

plt.xlabel("House Area (sq.ft.)")
plt.ylabel("House Price (Lakh)")
plt.title("House Area vs House Price")
plt.legend()
plt.grid()

plt.show()

# 16. GRAPH 2 - COST REDUCTION

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, iterations + 1),
    cost_history
)

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent Cost Reduction")
plt.grid()

plt.show()

# 17. GRAPH 3 - ACTUAL VS PREDICTED

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    gd_predictions
)

plt.xlabel("Actual Price (Lakh)")
plt.ylabel("Predicted Price (Lakh)")
plt.title("Actual Price vs Predicted Price")
plt.grid()

plt.show()

# 18. FINAL OBSERVATION

print("\n-----------------------------------------")
print("             FINAL OBSERVATION")
print("-------------------------------------------")

print(
    "Gradient Descent successfully optimized "
    "the Linear Regression parameters."
)

print(
    "The model performance was evaluated using "
    "MAE, MSE, RMSE and R2 score."
)

print(
    "The cost decreased during the iterations, "
    "showing that the model learned from the data."
)

print(
    "The Gradient Descent model was also compared "
    "with Scikit-learn Linear Regression."
)

print("\nProgram completed successfully.")
