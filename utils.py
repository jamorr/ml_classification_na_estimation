import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_classification_dataset(number: int = 1) -> tuple[pd.DataFrame, ...]:
    if number in (1,3):
        sep = "\t"
    else:
        sep = None
    train = read_missing(f"./classify/TrainData{number}.txt", sep=sep)
    target = read_missing(f"./classify/TrainLabel{number}.txt")
    test = read_missing(f"./classify/TestData{number}.txt", sep=sep)
    return train, target, test


def features_histograms_mean_std(data: pd.DataFrame) -> None:
    stats = data.describe().T
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    sns.histplot(stats["mean"], ax=ax1)
    sns.histplot(stats["std"], ax=ax2)


def read_missing(path: str, sep:str|None="\t", missing: list[str] = ["1.000000e+99"]) -> pd.DataFrame:
    if sep is not None:
        return pd.read_csv(path, header=None, sep=sep, na_values=missing)
    else:
        return pd.read_csv(path, header=None, delim_whitespace=True, na_values=missing)

