# Logistic Regression
This repository contains an implementation of logistic regression in Python. Logistic regression is a widely-used statistical method for predicting binary outcomes.

## Installation
To use the code in this repository, you will need to have Python 3 installed on your machine. You will also need to install the following libraries:

* NumPy
* pandas
* scikit-learn
You can install these libraries using pip by running the following command:

```
pip install numpy pandas scikit-learn
```

## Usage
To use the logistic regression model, you will need to have a dataset with a binary outcome variable and one or more predictor variables. The model can be trained and evaluated using the provided LogisticRegression class.

Here is an example of how to train and evaluate a logistic regression model on the pima-indians-diabetes dataset:

```Python3
from logistic_regression import LogisticRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = load_diabetes(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = model.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
```
## References
[Introduction to Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)

[Logistic Regression in Python](https://towardsdatascience.com/logistic-regression-a-simplified-approach-using-python-c4bc81a87c31)
