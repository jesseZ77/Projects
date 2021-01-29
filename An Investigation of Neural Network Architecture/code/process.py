import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    col_names = ["id",
                 "jitter local", "jitter local abs", "jitter rap", "jitter ppq5", "jitter ddp",
                 "shimmer local", "shimmer local db", "shimmer apq3", "shimmer apq5", "shimmer apq11", "shimmer dda",
                 "ac", "nth", "htn",
                 "median pitch", "mean pitch", "standard deviation of pitch", "minimum pitch", "maximum pitch",
                 "number of pulses", "number of periods", "mean period", "standard deviation of period",
                 "fraction of locally unvoiced frames", "number of voice breaks", "degree of voice breaks",
                 "updrs",
                 "class"]

    data_raw = pd.read_csv("data/raw_data.txt", header=None)
    data_raw.columns = col_names
    data = data_raw.drop(columns=["id", "updrs"])
    data.to_csv("data/data.csv", index=False)

    X = data.drop("class", axis=1).values
    y = data["class"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    np.savetxt("data/X_train.txt", X_train)
    np.savetxt("data/y_train.txt", y_train)
    np.savetxt("data/X_val.txt", X_val)
    np.savetxt("data/y_val.txt", y_val)
    np.savetxt("data/X_test.txt", X_test)
    np.savetxt("data/y_test.txt", y_test)


if __name__ == "__main__":
    main()
