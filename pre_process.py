from sklearn.preprocessing import LabelEncoder


def under_sample(df):

    df_agent = df[df["l1"] == "Agent"].sample(frac=0.92)
    df_place = df[df["l1"] == "Place"].sample(frac=0.79)
    df_species = df[df["l1"] == "Species"].sample(frac=0.54)
    df_work = df[df["l1"] == "Work"].sample(frac=0.54)
    df_event = df[df["l1"] == "Event"].sample(frac=0.48)

    df_new = df.copy()

    df_new = df_new.drop(index=df_agent.index)
    df_new = df_new.drop(index=df_place.index)
    df_new = df_new.drop(index=df_species.index)
    df_new = df_new.drop(index=df_work.index)
    df_new = df_new.drop(index=df_event.index)

    return df_new


def encode_label(df_train, df_test, df_val, source_label, new_label_name):
    LE = LabelEncoder()
    
    df_train[new_label_name] = LE.fit_transform(df_train[source_label])
    # breakpoint()
    df_test[new_label_name] = LE.transform(df_test[source_label])
    
    df_val[new_label_name] = LE.transform(df_val[source_label])
    
    return df_train, df_test, df_val
