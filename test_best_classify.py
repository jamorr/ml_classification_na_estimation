import joblib
import utils
from sklearn.model_selection import cross_val_score

def test_model_on_data(model, data_set_number:int):
    try:
        train, target, test = utils.read_classification_dataset(data_set_number)
        X, y = train.values, target.values.flatten()
        score = cross_val_score(model, X, y,scoring='f1_macro', cv=5)
        print(f"dataset {data_set_number} weighted F1 score 5 fold cv: ",score)
        score.sort()
        print(f"mean: {score.mean():.3f}", )
        print(f"median: {score[len(score)//2+1]:.3f}", )
        print(f"std: {score.std():.3f}", )
        predicitons = model.fit(X, y).predict(test.values)
        return predicitons

    except ValueError as e:
        print(e)


for i in range(1,6):
    model = joblib.load(f"./models/c_{i}_best_model.pkl")
    test_model_on_data(model, i)
