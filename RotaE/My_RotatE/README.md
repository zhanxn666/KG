
# RotaE/My_RotatE/RotatE_pykeen_pred.py and RotaE/My_RotatE/RotatE_pykeen_train.py

This script is used for generating predictions and training a pre-trained RotatE model using PyKEEN.

## Environment Configuration

Before running this script, you need to ensure that the following dependencies are installed:

- PyTorch
- PyKEEN
- pandas
- torch

You can install PyKEEN using pip:

```bash
pip install pykeen
```

## Deployment Requirements

To run this script, you need to have the following files:

- `TRAIN_PATH`: The path to the training dataset. This should be a tab-separated file with the format `head\trelation\ttail`.
- `TEST_PATH`: The path to the test dataset. This should also be a tab-separated file with the format `head\trelation\ttail`.
- `MODEL_DIR`: The directory where the pre-trained RotatE model is stored. The model should be saved as a pickle file named `trained_model.pkl`.
- `PRED_OUTPUT`: The path to the output file where the predictions will be saved. This should be a tab-separated file with the format `head\trelation\ttail\tpred`.

## Running the Script

To run the script, you can use the following command:

```bash
python RotatE_pykeen_pred.py
```

This will generate predictions using the pre-trained RotatE model and save the results to the specified output file.

To train the model, you can use the following command:

```bash
python RotatE_pykeen_train.py
```

This will train the RotatE model using the specified dataset and save the results to the specified output directory.

## Output

The output file will contain the predicted values for each triple in the test dataset. The format of the output file is `head\trelation\ttail\tpred`, where `tpred` is the predicted value for the triple.

The training process will output the loss and evaluation metrics at each epoch, as well as the final model parameters and evaluation metrics.

