import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def main():
    try:
        data = pd.read_csv('data.csv')
        X = data.km.values
        y = data.price.values

        n = len(X)

        normalized_X = X / np.linalg.norm(X)
        normalized_y = y / np.linalg.norm(y)

        theta0 = 0
        theta1 = 0

        lr = 0.5
        for _ in range(1000):
            sigma_theta0 = 0
            sigma_theta1 = 0
            for i in range(n):
                sigma_theta0 += (theta0 + theta1 * normalized_X[i]) - normalized_y[i]
                sigma_theta1 += ((theta0 + theta1 * normalized_X[i]) - normalized_y[i]) * normalized_X[i]
            theta0 -= lr * (1 / n) * sigma_theta0
            theta1 -= lr * (1 / n) * sigma_theta1

        formula = (theta0 + theta1 * normalized_X) * np.linalg.norm(y)
        plt.scatter(X, y)
        plt.plot(X, formula, c='r')
        plt.show()

        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()