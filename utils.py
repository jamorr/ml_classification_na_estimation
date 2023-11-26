import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os

def read_classification_dataset(number: int = 1) -> tuple[pd.DataFrame, ...]:
    path_to_data = pathlib.Path(os.path.dirname(__file__))
    print(path_to_data)
    match number:
        case 1:
            sep1 = sep2 = "\t"
        case 3:
            sep1 = "\t"
            sep2 = ","
        case _:
            sep1 = sep2 = None
    train = read_missing(path_to_data/"classify"/f"TrainData{number}.txt", sep=sep1)
    target = read_missing(path_to_data/"classify"/f"TrainLabel{number}.txt")
    test = read_missing(path_to_data/"classify"/f"TestData{number}.txt", sep=sep2)
    return train, target, test


def features_histograms_mean_std(data: pd.DataFrame) -> None:
    stats = data.describe().T
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(10, 10)
    sns.histplot(stats["mean"], ax=ax1)
    sns.histplot(stats["std"], ax=ax2)


def read_missing(path: str | pathlib.Path, sep:str|None="\t", missing: list[str] = ["1.000000e+99"]) -> pd.DataFrame:
    if sep is not None:
        return pd.read_csv(path, header=None, sep=sep, na_values=missing)
    else:
        return pd.read_csv(path, header=None, delim_whitespace=True, na_values=missing)

