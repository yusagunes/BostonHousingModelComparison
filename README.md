
# Boston House Price Prediction: AutoGluon vs. Linear Regression

## Project Overview

This project aims to compare the performance of two different machine learning models, AutoGluon and Linear Regression, on the Boston House Price dataset. The primary objective is to evaluate and contrast the models based on their predictive accuracy and computational efficiency.

## Dataset

The dataset used for this project is the Boston House Price dataset, which contains information about housing in the Boston area. It includes features such as the crime rate, number of rooms, property tax rate, and more, along with the target variable, which is the median house price.

### Features

1. **CRIM**: Per capita crime rate by town.
2. **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft.
3. **INDUS**: Proportion of non-retail business acres per town.
4. **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
5. **NOX**: Nitric oxides concentration (parts per 10 million).
6. **RM**: Average number of rooms per dwelling.
7. **AGE**: Proportion of owner-occupied units built prior to 1940.
8. **DIS**: Weighted distances to five Boston employment centers.
9. **RAD**: Index of accessibility to radial highways.
10. **TAX**: Full-value property tax rate per $10,000.
11. **PTRATIO**: Pupil-teacher ratio by town.
12. **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
13. **LSTAT**: Percentage of lower status of the population.
14. **MEDV**: Median value of owner-occupied homes in $1000s (target variable).

## Models

### AutoGluon

AutoGluon is an open-source AutoML toolkit developed by Amazon Web Services (AWS). It is designed to automate the process of building machine learning models, including hyperparameter tuning, feature selection, and model ensembling.

### Linear Regression

Linear Regression is a fundamental statistical technique used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables.

## Implementation

### Preprocessing

1. Load the dataset.
2. Handle missing values (if any).
3. Normalize/standardize features if required.
4. Split the data into training and testing sets.

### Model Training

1. **AutoGluon**: Use the `AutoGluon` library to automatically build and train multiple models. The best-performing model is selected based on evaluation metrics.
2. **Linear Regression**: Use the `scikit-learn` library to fit a Linear Regression model on the training data.

### Evaluation

1. Evaluate the performance of both models on the testing set using Mean Squared Error (MSE) and R-squared (R²) metrics.
2. Compare the results to determine which model performs better in terms of predictive accuracy.

## Results

### AutoGluon Model
- **MSE**: 8.479722354911248
- **R² Score**: 0.8861982274352317

### Linear Regression Model
- **MSE**: 21.51744423117721
- **R² Score**: 0.7112260057484932

### Conclusion

The AutoGluon model outperformed the Linear Regression model in terms of both Mean Squared Error and R² Score. This indicates that AutoGluon provides a more accurate prediction of house prices compared to a simple Linear Regression model.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- autogluon
- matplotlib (for visualizations)

## Usage

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the notebook `BostonHousePriceComparison.ipynb` to reproduce the results.

## License

This project is licensed under the MIT License.
