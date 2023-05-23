from os import getcwd
from math import sqrt
from csv import reader, writer


path = getcwd()


def mean(values):
    return sum(values) / float(len(values))

def covariance(x, mean_x, y, mean_y):
    result = 0.0
    for i in range(len(x)):
        result += (x[i] - mean_x) * (y[i] - mean_y)
    return result

def variance(list, mean):
    return sum([(x - mean) ** 2 for x in list])

def coefficient(covar, var, mean_x, mean_y):
    b1 = covar / var
    b0 = mean_y - (b1 * mean_x)
    return b1, b0

def load_data(dataset):
    init = 0
    x = list()
    y = list()
    with open(dataset) as file:
        content = reader(file)
        for row in content:
            if init == 0:
                init = 1
            else:
                x.append(row[0])
                y.append(row[1])
    return x, y

def split_dataset(x, y):
    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()

    training_size = int(.8 * len(x))

    x_train, x_test = x[0:training_size], x[training_size::]
    y_train, y_test = y[0:training_size], y[training_size::]

    return x_train, y_train, x_test, y_test

# Function to calculate the equation: y = B1 * x + B0
def predict(b0, b1, test_X):
    predicted_y = list()
    for i in test_X:
        predicted_y.append(b0 + b1 * i)
    return predicted_y

# Function to calculate RMSE
def rmse(predicted_y, test_y):
    sum_error = 0.0
    for i in range(len(predicted_y)):
        error = predicted_y[i] - test_y[i]
        sum_error += error ** 2
    mean_squared_error = sum_error / float(len(test_y))
    result = sqrt(mean_squared_error)
    return result


def main():
    try:
        # Load dataset
        dataset = path + '/dataset.csv'
        x, y = load_data(dataset)

        # Prepare data
        x = [float(i) for i in x]
        y = [float(i) for i in y]

        # Calculate average value of x and y, covariance and variance
        mean_x = mean(x)
        mean_y = mean(y)
        covar = covariance(x, mean_x, y, mean_y)
        var = variance(x, mean_x)

        # Split data in x and y
        x_train, y_train, x_test, y_test = split_dataset(x, y)

        # Calculate the coefficient
        b1, b0 = coefficient(covar, var, mean_x, mean_y)

        print('')
        print('Coefficients: ')
        print('B1: ', b1)
        print('B0: ', b0)

        # Prediction using the model
        predicted_y = predict(b0, b1, x_test)

        # Error of the model
        mean_error = rmse(predicted_y, y_test)

        print('')
        print('Linear regression model without framework')
        print('Root mean squared error of the model: {}'.format(mean_error))

        # Using the model to predict new data
        new_data = int(input('Type the value for the investment: '))

        # Predict
        new_y = b0 + b1 * new_data
        print('Predicted ROI: {}'.format(new_y))

    except Exception as error:
        print(error)

main()