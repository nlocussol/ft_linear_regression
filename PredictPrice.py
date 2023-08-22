import pandas as pd
import os

def main():
    try:
        if os.path.isfile("./dataset/theta.csv"):
            theta_data = pd.read_csv("./dataset/theta.csv")
            first_row = theta_data.iloc[0]
            theta0 = first_row['theta0']
            theta1 = first_row['theta1']
        else:
            theta0, theta1 = 0, 0
            
        mileage = input("Enter a mileage (in km): ") # handle negative
        if (float(mileage) < 0):
            raise Exception("mileage can't be negative!")
        print(f"This car worth {theta0 + theta1 * float(mileage):.0f} euro")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()