import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

def update_theta(theta0, theta1):
    with open('./dataset/theta.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["theta0", "theta1"]
        writer.writerow(field)
        writer.writerow([theta0, theta1])

def normalized_data():
    data = pd.read_csv('./dataset/data.csv')
    X = data.km.values
    y = data.price.values

    normalized_X = X / np.linalg.norm(X)
    normalized_y = y / np.linalg.norm(y)

    return normalized_X, normalized_y, np.linalg.norm(X), np.linalg.norm(y)

def gradient_descent(X, y, norm_X, norm_y, lr=0.5, iteration=1000):
    theta0 = 0
    theta1 = 0
    n = len(X)

    for _ in range(iteration):
        sigma_theta0 = 0
        sigma_theta1 = 0

        for i in range(n):
            sigma_theta0 += (theta0 + theta1 * X[i]) - y[i]
            sigma_theta1 += ((theta0 + theta1 * X[i]) - y[i]) * X[i]

        theta0 -= lr * (1 / n) * sigma_theta0
        theta1 -= lr * (1 / n) * sigma_theta1
        update_theta(theta0 * norm_y, theta1 * norm_X)

    theta0 = theta0 * norm_y
    theta1 = theta1 * (norm_y / norm_X)
    update_theta(theta0, theta1)
    return theta0, theta1

def show_graph(X, y, linear_reg):
    plt.scatter(X, y)
    plt.plot(X, linear_reg, c='r')
    plt.xlabel('mileage (in km)')
    plt.ylabel('price (in euro)')
    plt.title('Car price by kilometer')
    plt.show()

def coef_determination(y, pred):
    total_pred_diff = sum((y - pred) ** 2)
    total_mean = sum((y - y.mean()) ** 2)
    return 1 - total_pred_diff / total_mean

def main():
    try:
        X, y, norm_X, norm_y = normalized_data()
        denorm_X = X * norm_X
        denorm_y = y * norm_y
        theta0, theta1 = gradient_descent(X, y, norm_X, norm_y)
        linear_reg = theta0 + theta1 * denorm_X

        algo_precision = coef_determination(denorm_y, linear_reg)
        print(f"precision: {algo_precision:.2f}%")
        show_graph(denorm_X, denorm_y, linear_reg)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()