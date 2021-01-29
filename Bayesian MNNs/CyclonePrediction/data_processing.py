import numpy as np
import pickle


def split_data(data, window=4):
    X_list = []
    y_list = []

    for cyclone in data:
        n = len(cyclone)
        x = np.array([cyclone[i:(i + window), 1:3] for i in range(n - window)])
        y = np.array([cyclone[i + window, 1:3] for i in range(n - window)])

        X_list.append(x)
        y_list.append(y)

    return X_list, y_list


def process_data():
    for dataset in ['train', 'val', 'test']:
        with open(f"data/{dataset}_raw.pkl", "rb") as r:
            data = pickle.load(r)

        X_list, y_list = split_data(data)

        with open(f"data/X_split_{dataset}.pkl", "wb") as f:
            pickle.dump(X_list, f)

        with open(f"data/y_split_{dataset}.pkl", "wb") as f:
            pickle.dump(y_list, f)

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        np.save(f"data/X_{dataset}.npy", X)
        np.save(f"data/y_{dataset}.npy", y)

    print("Done")


if __name__ == "__main__":
    process_data()
