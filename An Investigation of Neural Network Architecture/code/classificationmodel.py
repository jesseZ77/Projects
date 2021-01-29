from class_model import ClassificationModel
import numpy as np
import pandas as pd
import re
from textwrap import wrap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({'font.size': 18})


def main():
    # load data
    data = pd.read_csv("data/data.csv")
    X_train = np.loadtxt("data/X_train.txt")
    y_train = np.loadtxt("data/y_train.txt")
    X_val = np.loadtxt("data/X_val.txt")
    y_val = np.loadtxt("data/y_val.txt")
    X_test = np.loadtxt("data/X_test.txt")
    y_test = np.loadtxt("data/y_test.txt")

    # data_visualisation(data)
    evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, test_only=True)
    # evaluate_best_model(X_train, y_train, X_val, y_val, X_test, y_test)
    # evaluate_logistic_model(X_train, y_train, X_val, y_val, X_test, y_test)


def data_visualisation(data):
    features = [col for col in data.columns if col.lower != "class"]

    # data exploration: feature summary statistics
    summary_stats = data.drop("class", axis=1).describe().T[["mean", "std", "min", "25%", "75%", "max"]]
    table = (summary_stats
             .applymap(lambda x: str(int(x)) if abs(x - int(x)) < 1e-6 else str(round(x, 2)))
             .to_latex(caption="Summary statistics",
                       column_format='p{1.75cm}|p{0.60cm}|p{0.60cm}|p{0.60cm}|p{0.60cm}|'
                                     'p{0.60cm}|p{0.60cm}|p{0.60cm}'))
    table = re.sub("\\\\begin{table}", "\\\\begin{table}[H]", table)
    text_file = open(f'Tables/Summary statistics.txt', "w")
    text_file.write(table)
    text_file.close()

    # data exploration: feature density plots
    for k in range(2):
        fig = plt.figure(figsize=(9, 9))
        for i in range(9):
            col = features[i + 9 * k]
            plt.subplot(3, 3, i + 1)
            plt.hist(data[col], density=True)
            plt.title("\n".join(wrap(col, 20)), fontsize=18)

        plt.tight_layout()
        fig.savefig(f"Images/density plots {9 * k + 1}-{9 * (k + 1)}.pdf")
        plt.close()

    fig = plt.figure(figsize=(9, 9))
    for i in range(8):
        col = features[i + 18]
        plt.subplot(3, 3, i + 1)
        plt.hist(data[col], density=True)
        plt.title("\n".join(wrap(col, 20)), fontsize=18)
    plt.tight_layout()
    fig.savefig(f"Images/density plots {19}-{26}.pdf")
    plt.close()

    # data exploration: target
    class_distribution = data.groupby("class").agg({"class": "count"}).rename(columns={"class": "count"})
    table = class_distribution.to_latex(caption="Class distribution")
    table = re.sub("\\\\begin{table}", "\\\\begin{table}[H]", table)
    text_file = open(f'Tables/Class distribution.txt', "w")
    text_file.write(table)
    text_file.close()

    fig = plt.figure(figsize=(8, 4))
    plt.title("Target distribution")
    plt.hist(data["class"], density=True)
    fig.savefig(f"Images/density plots target.pdf")
    plt.close()


def evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, test_only):
    # test
    clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test), n_reps=10,
                              caption="test")
    r, results, results_summary = clm.evaluate()
    print(results_summary)

    if test_only:
        return 0

    # SGD vs ADAM
    for optim in ["sgd", "adam"]:
        clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test),
                                  caption=optim, optim=optim)
        clm.evaluate()

    # Learning rate
    for lr in [0.10, 0.50, 1.00]:
        clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test),
                                  caption=f"Learning rate = {lr}", lr=lr)
        clm.evaluate()

    # Momentum
    for mom in [0.01, 0.10, 0.50, 1.00]:
        clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test),
                                  caption=f"Momentum = {mom}", mom=mom)
        clm.evaluate()


    # Number of hidden neurons
    for n_hidden_neurons in [5, 15, 20, 25]:
        clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test),
                                  caption=f"Hidden neurons = {n_hidden_neurons}", n_hidden_neurons=n_hidden_neurons)
        clm.evaluate()

    # Number of hidden layers
    for n_hidden_layers in [2, 3, 4]:
        clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test),
                                  caption=f"Hidden layers = {n_hidden_layers}",
                                  n_hidden_layers=n_hidden_layers, n_hidden_neurons=25)
        clm.evaluate()


def evaluate_best_model(X_train, y_train, X_val, y_val, X_test, y_test):
    clm = ClassificationModel((X_train, y_train), (X_val, y_val), (X_test, y_test),
                              caption=f"Best model",
                              optim="sgd",
                              lr=0.01, mom=0.00,
                              n_hidden_layers=1, n_hidden_neurons=25, n_reps=1)
    r, results, _ = clm.evaluate(write=False)
    print(results)
    y_pred_train = clm.model.predict(X_train).reshape(-1)
    y_pred_val = clm.model.predict(X_val).reshape(-1)
    y_pred_test = clm.model.predict(X_test).reshape(-1)
    write_table(results.T.rename({0: "Metrics"}), "Best model metrics", "best model metrics")

    # training plots
    plot_training_graphs(clm, r, results)

    # confusion matrices
    conf_mat_train = my_confusion_matrix(y_train, y_pred_train)
    conf_mat_val = my_confusion_matrix(y_val, y_pred_val)
    conf_mat_test = my_confusion_matrix(y_test, y_pred_test)

    write_table(conf_mat_train, caption="Confusion matrix: train", file_name="confusion_matrix_network_train")
    write_table(conf_mat_val, caption="Confusion matrix: train", file_name="confusion_matrix_network_val")
    write_table(conf_mat_test, caption="Confusion matrix: train", file_name="confusion_matrix_network_test")

    df_summary_table = summary_table(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
    write_table(df_summary_table, caption="Evaluation metrics", file_name="eval_metrics_network")



    # ROC and AUC
    plot_roc(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, "network")

    # Precision, recall f1 score
    plot_precision_recall(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, "network")


def evaluate_logistic_model(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    logreg = LogisticRegression()
    logreg.fit(X_train_val, y_train_val)

    y_pred_train = logreg.predict_proba(X_train)[:, 1]
    y_pred_val = logreg.predict_proba(X_val)[:, 1]
    y_pred_test = logreg.predict_proba(X_test)[:, 1]

    # confusion matrices
    conf_mat_train = my_confusion_matrix(y_train, y_pred_train)
    conf_mat_val = my_confusion_matrix(y_val, y_pred_val)
    conf_mat_test = my_confusion_matrix(y_test, y_pred_test)

    write_table(conf_mat_train, caption="Confusion matrix: train", file_name="confusion_matrix_log_reg_train")
    write_table(conf_mat_val, caption="Confusion matrix: train", file_name="confusion_matrix_log_reg_val")
    write_table(conf_mat_test, caption="Confusion matrix: train", file_name="confusion_matrix_log_reg_test")

    df_summary_table = summary_table(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test)
    write_table(df_summary_table, caption="Evaluation metrics", file_name="eval_metrics_log_reg")

    # ROC and AUC
    plot_roc(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, "logreg")

    # Precision, recall f1 score
    plot_precision_recall(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, "logreg")


def plot_training_graphs(clm, r, results):
    # Loss and accuracy per iteration, results
    fig = plt.figure(figsize=(6, 12))

    plt.subplot(2, 1, 1)
    plt.title(f"Best model: Loss per iteration")
    plt.plot(r.history['loss'], label='Loss')
    plt.plot(r.history['val_loss'], label='Validation Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f"Best model: Accuracy per iteration")
    plt.plot(r.history['accuracy'], label='Accuracy')
    plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    fig.savefig(f"Best Model Eval/Best model.pdf")
    plt.close()


def write_table(df, caption, file_name):
    table = (df
             .applymap(lambda x: str(int(x)) if abs(x - int(x)) < 1e-6 else str(round(x, 4)))
             .to_latex(caption=caption))
    table = re.sub("\\\\begin{table}", "\\\\begin{table}[H]", table)
    text_file = open(f'Best Model Eval/{file_name}.txt', "w")
    text_file.write(table)
    text_file.close()


def plot_roc(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, caption):
    roc_train = roc_curve(y_train, y_pred_train)
    roc_val = roc_curve(y_val, y_pred_val)
    roc_test = roc_curve(y_test, y_pred_test)

    fig = plt.figure(figsize=(8, 6))
    plt.title(f"ROC curve \n"
              f"{caption}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(roc_train[0], roc_train[1], label="train")
    plt.plot(roc_val[0], roc_val[1], label="validation")
    plt.plot(roc_test[0], roc_test[1], label="test")
    plt.legend()
    fig.savefig(f"Best Model Eval/ROC_{caption}.pdf")
    plt.close()


def plot_precision_recall(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, caption):
    pr_train = precision_recall_curve(y_train, y_pred_train)
    pr_val = precision_recall_curve(y_val, y_pred_val)
    pr_test = precision_recall_curve(y_test, y_pred_test)

    fig = plt.figure(figsize=(8, 6))
    plt.title(f"Precision vs recall curve \n"
              f"{caption}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(pr_train[0], pr_train[1], label="train")
    plt.plot(pr_val[0], pr_val[1], label="validation")
    plt.plot(pr_test[0], pr_test[1], label="test")
    plt.legend()
    fig.savefig(f"Best Model Eval/Precision recall_{caption}.pdf")
    plt.close()


def my_confusion_matrix(y_true, y_pred):
    df_conf_mat = pd.DataFrame(confusion_matrix(y_true, np.where(y_pred > 0.5, 1, 0)))
    df_conf_mat.rename(columns={0: "Pred negative", 1: "Pred positive"},
                       index={0: "Actual negative", 1: "Actual positive"}, inplace=True)
    df_conf_mat.loc["Total pred"] = df_conf_mat.sum(axis=0)
    df_conf_mat["Total actual"] = df_conf_mat.sum(axis=1)
    return df_conf_mat


def tpr(y_true, y_pred):
    confusion_mat = my_confusion_matrix(y_true, y_pred)
    return (confusion_mat.loc["Actual positive", "Pred positive"] /
            confusion_mat.loc["Actual positive", "Total actual"])


def fpr(y_true, y_pred):
    confusion_mat = my_confusion_matrix(y_true, y_pred)
    return (confusion_mat.loc["Actual negative", "Pred positive"] /
            confusion_mat.loc["Actual negative", "Total actual"])


def precision(y_true, y_pred):
    confusion_mat = my_confusion_matrix(y_true, y_pred)
    return (confusion_mat.loc["Actual positive", "Pred positive"] /
            confusion_mat.loc["Total pred", "Pred positive"])


def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_true, y_pred):
    return np.mean(y_true == np.where(y_pred > 0.5, 1, 0))


def my_f1_score(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = tpr(y_true, y_pred)
    return 2 * precision_ * recall_ / (precision_ + recall_)


def summary_table(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test):
    df = pd.DataFrame({
        "loss": [log_loss(y_train, y_pred_train), log_loss(y_val, y_pred_val), log_loss(y_test, y_pred_test)],
        "accuracy": [accuracy(y_train, y_pred_train), accuracy(y_val, y_pred_val), accuracy(y_test, y_pred_test)],
        "tpr/ recall": [tpr(y_train, y_pred_train), tpr(y_val, y_pred_val), tpr(y_test, y_pred_test)],
        "fpr": [fpr(y_train, y_pred_train), fpr(y_val, y_pred_val), fpr(y_test, y_pred_test)],
        "precision": [precision(y_train, y_pred_train), precision(y_val, y_pred_val), precision(y_test, y_pred_test)],
        "f1 score": [my_f1_score(y_train, y_pred_train), my_f1_score(y_val, y_pred_val), my_f1_score(y_test, y_pred_test)],
        "auc": [roc_auc_score(y_train, y_pred_train), roc_auc_score(y_val, y_pred_val), roc_auc_score(y_test, y_pred_test)]},
        index=["train", "val", "test"]).T

    return df


if __name__ == "__main__":
    main()
