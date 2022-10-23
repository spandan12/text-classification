import pandas as pd

# Read training data file
def read_training_data():

    return pd.read_csv("under_sampled_train_data.csv")


# Read test data file
def read_test_data():

    return pd.read_csv("DBPEDIA_test.csv")


# Read validation data file
def read_val_data():

    return pd.read_csv("DBPEDIA_val.csv")
