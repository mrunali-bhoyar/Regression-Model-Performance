# Performance Evaluation of Regression Model

## Practical Assignment-03

### Problem Statement

1. Develop Regression Models for a real-world application and evaluate their performance using appropriate metrics.
2. Implement and analyze the performance of Gradient Descent optimization for Linear Regression.

## Application

**House Price Prediction**

The project predicts house prices based on the area of the house in square feet. Linear Regression is used because there is a relationship between the house area and its price.

## Objectives

- Develop a Linear Regression model for house price prediction.
- Implement Gradient Descent manually for Linear Regression.
- Evaluate the model using MAE, MSE, RMSE and R² score.
- Compare the manually implemented Gradient Descent model with Scikit-learn Linear Regression.
- Predict the price of a new house based on its area.

## Dataset

The project uses a small sample dataset containing:

- **Area:** House area in square feet.
- **Price:** House price in lakh.

The dataset is created directly inside the Python program, so no separate dataset file is required.

## Technologies and Libraries

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Implementation

The program performs the following steps:

1. Creates the house price dataset.
2. Separates input and output variables.
3. Splits the data into training and testing sets.
4. Takes learning rate and number of iterations as user inputs.
5. Standardizes the input data.
6. Implements Gradient Descent manually.
7. Calculates predictions.
8. Evaluates the model using MAE, MSE, RMSE and R² score.
9. Compares the result with Scikit-learn Linear Regression.
10. Predicts the price of a new house.
11. Displays graphs for regression, cost reduction and actual versus predicted values.

## How to Run

The program was developed and tested using Google Colab.

1. Open Google Colab.
2. Upload or paste `regression_model.py` into a notebook.
3. Install the required libraries if needed.
4. Run the program.
5. Enter the learning rate, number of iterations and new house area when prompted.

The required Python libraries are listed in `requirements.txt`.

## Model Evaluation

The model is evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

## Sample Result

For the test run using:

- Learning Rate = 0.01
- Iterations = 1000
- New House Area = 1500 sq.ft.

the program produced:

- MAE = 1.1673
- MSE = 1.6054
- RMSE = 1.2671
- R² Score = 0.9987
- Predicted House Price = 79.09 lakh

These results are based on the sample dataset used in this practical.

## Project Files

```text
Regression-Model-Performance/
│
├── regression_model.py
├── requirements.txt
└── README.md
```

## Conclusion

The project demonstrates Linear Regression for house price prediction and manually implements Gradient Descent to optimize the model parameters. The model is evaluated using standard regression metrics and compared with the Scikit-learn implementation.
