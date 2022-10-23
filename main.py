from files import read_training_data, read_test_data, read_val_data
from pre_process import under_sample, encode_label
from embeddings import create_embedding
from models import bert_model_and_tokenizer
from classifications import logisticRegression
import numpy as np


def main():
    print('checkpoint 1: Loading data')
    df_train = read_training_data()
    df_test = read_test_data()
    df_val = read_val_data()

    # print('checkpoint 2: Loading tokenizer and model')
    # device, tokenizer, model = bert_model_and_tokenizer()

    # print('checkpoint 3: Preprocessing data')
    # under_sampled_df = under_sample(df=df_train)
    
    df_train, df_test, df_val = encode_label(
        df_train=df_train,
        df_test=df_test,
        df_val=df_val,
        source_label="l1",
        new_label_name="label_l1",
    )
    
    
    # print('checkpoint 4 train: creating embeddings')
    # train_embeddings = create_embedding(
    #     df=df_train, tokenizer=tokenizer, model=model, device=device
    # )
    # np.save("train_embeddings.npy", train_embeddings)
    # print('checkpoint 4 train: creating embeddings')
    # test_embeddings = create_embedding(
    #     df=df_test, tokenizer=tokenizer, model=model, device=device
    # )
    # np.save("test_embeddings.npy", test_embeddings)
    
    print('loading embeddings')
    train_embeddings = np.load('under_sampled_embeddings.npy')
    test_embeddings = np.load('test_embeddings.npy')
    
    print('checkpoint 5: getting labels')
    y_train = df_train["label_l1"]
    y_test = df_test["label_l1"]

    print('checkpoint 6: Running the classification model')
    logisticRegression(
        x_train=train_embeddings, x_test=test_embeddings, y_train=y_train, y_test=y_test
    )


if __name__ == "__main__":
    main()
