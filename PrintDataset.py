import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        data = pd.read_csv("./data.csv")
        plt.scatter(data.km, data.price)
        plt.xlabel('km')
        plt.ylabel('price')
        plt.title('Car price by kilometer')
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()