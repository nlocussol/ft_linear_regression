import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    try:
        data = pd.read_csv("./data.csv")
        mileage = data.km.values
        price = data.price.values

        # Normalize the data
        normalized_mileage = (mileage - np.mean(mileage)) / np.std(mileage)
        normalized_price = (price - np.mean(price)) / np.std(price)

        n = len(normalized_mileage)
        learning_rate = 0.01
        iterations = 1000

        theta0, theta1 = 0, 0

        # Gradient Descent
        for _ in range(iterations):
            gradient_theta0 = 0
            gradient_theta1 = 0
                    
            for i in range(n):
                prediction = theta0 + theta1 * normalized_mileage[i]
                gradient_theta0 += prediction - normalized_price[i]
                gradient_theta1 += (prediction - normalized_price[i]) * normalized_mileage[i]
                    
            theta0 -= learning_rate * (1/n) * gradient_theta0
            theta1 -= learning_rate * (1/n) * gradient_theta1

        # Denormalize the parameters
        denorm_theta0 = (np.mean(price) - theta1 * np.mean(mileage)) / np.std(mileage)
        denorm_theta1 = theta1 * np.std(price) / np.std(mileage)

        # Denormalize the predicted prices
        denorm_predicted_prices = denorm_theta0 + denorm_theta1 * mileage

        # Plotting
        plt.scatter(mileage, price, label='Data')
        plt.plot(mileage, denorm_predicted_prices, color='red', label='Regression Line')
        plt.xlabel('Kilometers')
        plt.ylabel('Price')
        plt.legend()
        plt.title('Linear Regression with Gradient Descent (Denormalized)')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()