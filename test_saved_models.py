import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from utils import read_classification_dataset


# Function to save predictions to a text file
def save_predictions(predictions, model_name):
    output_filename = f"./predictions/{model_name}_predictions.txt"
    np.savetxt(output_filename, predictions, fmt="%d")
    print(f"Predictions saved to {output_filename}")

# Main script
models_folder = "models"
if not os.path.exists(models_folder):
    print(f"The folder '{models_folder}' does not exist.")
else:
    for filename in os.listdir(models_folder):
        if filename.endswith(".pkl"):
            model_char = int(filename[2])
            train, label, test = read_classification_dataset(model_char)

            # Load the model
            model_path = os.path.join(models_folder, filename)
            model = joblib.load(model_path)

            # Make predictions on the test set
            predictions = model.predict(test.values)
            print(cross_val_score(model,X=train.values, y=label.values.flatten(), scoring='f1_macro'))
            # Save predictions to a text file
            save_predictions(predictions, filename[:-4])  # Remove '.pkl' extension from filename
