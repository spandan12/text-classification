from files import read_training_data, read_test_data
from pre_process import under_sample, encode_label
from embeddings import create_embedding
from models import bert_model_and_tokenizer
from classifications import logisticRegression


def main():
    df_train = read_training_data()
    df_test = read_test_data()

    device, tokenizer, model = bert_model_and_tokenizer()

    under_sampled_df = under_sample(df=df_train)
    train_embeddings = create_embedding(
        df=under_sampled_df, tokenizer=tokenizer, model=model, device=device
    )

    test_embeddings = create_embedding(
        df=df_test, tokenizer=tokenizer, model=model, device=device
    )

    under_sampled_df, df_test = encode_label(
        df_train=under_sampled_df,
        df_test=df_test,
        source_label="l1",
        new_label_name="label_l1",
    )

    y_train = under_sampled_df["label_l1"]
    y_test = df_test["label_l1"]

    logisticRegression(
        x_train=train_embeddings, x_test=test_embeddings, y_train=y_train, y_test=y_test
    )


if __name__ == "__main__":
    main()
