import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_classification_dataset(number:int = 1)->tuple[pd.DataFrame, ...]:
    train = pd.read_csv(
        f"./classify/TrainData{number}.txt",
        header=None, sep='\t',
        na_values=["1.000000e+99"]
        )
    target = pd.read_csv(
        f"./classify/TrainLabel{number}.txt",
        header=None,
        sep='\t'
        )
    test = pd.read_csv(
        f"./classify/TestData{number}.txt",
        header=None,
        sep='\t',
        na_values=["1.000000e+99"]
        )
    return train, target, test

def features_histograms_mean_std(data:pd.DataFrame)->None:
    stats = data.describe().T
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.set_size_inches(10,10)
    sns.histplot(stats['mean'], ax=ax1)
    sns.histplot(stats['std'], ax=ax2)