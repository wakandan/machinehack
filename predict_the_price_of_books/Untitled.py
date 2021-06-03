#!/usr/bin/env python
# coding: utf-8

# In[685]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from textblob import Word
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from livelossplot import PlotLossesKeras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from argparse import ArgumentParser
import tensorflow as tf
import os

# prepare to generate word embbeding vectors
glove_input_file = "/Users/kdang/Documents/glove.6B/glove.6B.100d.txt"
word2vec_output_file = "glove.6B.100d.txt.word2vec"

if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file)

# In[686]:


df = pd.read_excel("Participants_Data/Data_Train.xlsx")
df_test = pd.read_excel("Participants_Data/Data_Test.xlsx")


def process_df(df, target_file=None, is_test=False):

    # if target_file is not None and os.path.exists(target_file):
    #     print("already processed, return")
    #     df_processed = pd.read_csv(target_file)
    #     return df_processed
    # In[578]:

    len(df)

    # In[579]:

    df.head()

    # In[580]:

    df["RatingValue"] = df["Reviews"].apply(lambda x: float(x.split()[0]) / 5)

    # In[581]:

    df["NumReview"] = df["Ratings"].apply(lambda x: int(x.split()[0].replace(",", "")))

    # In[582]:

    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthsMaxDays = {
        "Jan": 31,
        "Feb": 28,
        "Mar": 31,
        "Apr": 30,
        "May": 31,
        "Jun": 30,
        "Jul": 31,
        "Aug": 31,
        "Sep": 30,
        "Oct": 31,
        "Nov": 30,
        "Dec": 31,
    }
    months_to_num = {}
    for i, m in enumerate(months):
        months_to_num[m] = i + 1
    months_to_num

    # In[583]:

    df.Edition = df.Edition.apply(lambda x: x.replace("–", "-").replace("-", "-"))
    df.loc[df.Edition.str.match(".*\(\w+\).*"), "Edition"] = df.loc[
        df.Edition.str.match(".*\(\w+\).*"), "Edition"
    ].apply(lambda x: re.sub("\(\w+\),", "", x))
    df.loc[
        df.Edition.str.match(r".*[^\d]+$"), "EditionDate"
    ] = np.nan  # pattern 1, year and month

    # In[584]:

    len(df)

    # In[585]:

    df.loc[df.Edition.str.match(r".*\d+$"), "EditionDate"] = df.Edition.apply(
        lambda x: x.split(",")[-1].split("-")[-1].strip()
    )

    # In[586]:

    df.EditionDate

    # In[587]:

    df["Year"] = df.EditionDate.apply(
        lambda x: int(x[-4:].strip()) if pd.notnull(x) else np.nan
    )
    df["Year"].unique()

    # In[588]:

    df["Month"] = df.EditionDate.apply(
        lambda x: x[-8:-4] if (pd.notnull(x) and len(x) > 8) else np.nan
    )

    # In[589]:

    df["Day"] = df.EditionDate.apply(
        lambda x: int(x[-11:-9].strip()) if (pd.notnull(x) and len(x) >= 10) else np.nan
    )

    # In[590]:

    sns.countplot(data=df, y="BookCategory")

    # In[591]:

    df["PrintEdition"] = df.Edition.apply(lambda x: x.split(",")[0])

    # In[592]:

    sns.countplot(data=df, y="PrintEdition")

    # In[593]:

    df.to_csv("processed_edition.csv")  # write to file to save time later

    # In[594]:

    # df.loc[df.Edition.str.contains('Import'), 'IsImported'] = 1 # mark books that were imported
    df["IsImported"] = df.Edition.str.contains("Import")

    # In[595]:

    df["IsBook"] = df.Genre.str.contains("(Books)")

    # In[596]:

    df["IsMultipleAuthor"] = df.Author.str.match(".*[,&-].*")

    # In[597]:

    sns.countplot(data=df, y="IsMultipleAuthor")

    # In[598]:

    df["IsSpecialAuthor"] = df.Author.apply(
        lambda x: any(
            i
            for i in x.lower().split()
            if i
            in [
                "phd.",
                "phd",
                "dr",
                "dr.",
                "prof",
                "prof.",
                "sir",
                "sir.",
                "m.d.",
                "m.d.",
                "mr",
                "mr.",
                "mrs",
                "mrs.",
                "m.a",
                "m.a.",
            ]
        )
    )

    # In[599]:

    sns.countplot(data=df, y="IsSpecialAuthor")

    # In[600]:

    sns.countplot(data=df, y="IsBook")

    # In[601]:

    # In[603]:

    def processTextColumn(df, col):
        df[f"{col}Token"] = df[col].str.lower()
        df[f"{col}Token"] = (
            df[f"{col}Token"]
            .str.replace("[^\w\s]", "")
            .apply(lambda x: [Word(i).lemmatize() for i in x.split()])
        )
        stop = stopwords.words("english")
        df[f"{col}Token"] = df[f"{col}Token"].apply(
            lambda x: [i for i in x if i not in stop]
        )
        df[f"{col}Vector"] = df[f"{col}Token"].apply(
            lambda x: np.mean(
                [(model[i] if i in model else np.zeros(100)) for i in x], axis=0
            )
            if x
            else np.zeros(100)
        )

    # In[604]:

    processTextColumn(df, "Synopsis")
    processTextColumn(df, "BookCategory")
    processTextColumn(df, "Title")
    processTextColumn(df, "Genre")
    df = df.drop(
        columns=["BookCategoryToken", "BookCategoryToken", "TitleToken", "GenreToken"]
    )

    # In[605]:

    min_year = df.Year.min()
    min_year
    max_year = df.Year.max()
    max_year
    random.randrange(min_year, max_year + 1)
    df.loc[pd.isnull(df.Year), "Year"] = df.loc[pd.isnull(df.Year), "Year"].apply(
        lambda x: random.randrange(min_year, max_year + 1)
    )

    # In[606]:

    df.Month.unique()

    # In[607]:

    df.loc[pd.isnull(df.Month), "Month"] = df.loc[pd.isnull(df.Month), "Month"].apply(
        lambda x: random.choice(months)
    )

    # In[608]:

    df.loc[pd.isnull(df.Day), "Day"] = df.loc[
        df.loc[pd.isnull(df.Day)].index, "Month"
    ].apply(lambda x: random.randint(1, monthsMaxDays[x]))

    # In[609]:

    sns.countplot(data=df, y="IsSpecialAuthor")

    # In[610]:
    if not is_test:
        bins = [0, 1000, 5000, 100000]
        df["PriceBinned"] = np.searchsorted(bins, df["Price"].values)

    df["IsImported"] = df["IsImported"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df["IsBook"] = df["IsBook"].astype(int)
    df["IsMultipleAuthor"] = df["IsMultipleAuthor"].astype(int)
    df["IsSpecialAuthor"] = df["IsSpecialAuthor"].astype(int)

    # In[627]:

    df.info()

    # In[668]:

    df_processed = df.copy()
    df_processed = df_processed.drop(
        columns="Title Author Edition Reviews Ratings Synopsis Genre BookCategory EditionDate SynopsisToken".split()
    )

    # In[669]:

    df_processed.info()

    # In[670]:

    df_processed["Year"] = MinMaxScaler().fit_transform([[i] for i in df.Year])
    df_processed["NumReview"] = MinMaxScaler().fit_transform(
        [[i] for i in df.NumReview]
    )

    # In[671]:

    def categorical_to_column(df, col, unique_values):
        # for i in df[col].unique():
        for i in unique_values:
            col_name = f"{col}_{i}"
            df.loc[df[col] == i, col_name] = 1
            df[col_name] = df[col_name].fillna(0)
            df[col_name] = df[col_name].astype(int)

    # In[672]:

    categorical_to_column(df_processed, "Month", months)
    categorical_to_column(df_processed, "Day", list(range(1, 32)))
    categorical_to_column(
        df_processed,
        "PrintEdition",
        [
            "Paperback",
            "Hardcover",
            "Mass Market Paperback",
            "Sheet music",
            "Flexibound",
            "Plastic Comb",
            "Loose Leaf",
            "Tankobon Softcover",
            "Perfect Paperback",
            "Board book",
            "Cards",
            "Spiral-bound",
            "Product Bundle",
            "Library Binding",
            "Leather Bound",
        ],
    )
    df_processed = df_processed.drop(columns=["Month", "Day", "PrintEdition"])

    # In[673]:

    for col in ["GenreVector", "TitleVector", "BookCategoryVector", "SynopsisVector"]:
        for i in range(100):
            df_processed[f"{col}{i}"] = df_processed[col].apply(lambda x: x[i])
    df_processed = df_processed.drop(
        columns=["GenreVector", "TitleVector", "BookCategoryVector", "SynopsisVector"]
    )
    df_processed.to_csv(target_file)
    print(f"wrote to file {target_file}")

    return df_processed


def train_mlp_sklearn(df_train_processed):
    xtrain, xval, ytrain, yval = train_test_split(
        df_train_processed.drop(columns=["Price"]),
        df_train_processed.Price,
        stratify=df_train_processed["PriceBinned"],
    )

    xtrain = xtrain.drop(columns=["PriceBinned"])
    xval = xval.drop(columns=["PriceBinned"])
    mlp = MLPRegressor(
        verbose=True,
        validation_fraction=0.1,
        max_iter=10000,
        tol=0.1,
        hidden_layer_sizes=(100, 50),
    )
    mlp.fit(xtrain, ytrain)
    ypred = mlp.predict(xval)
    metric = 1 - np.sqrt(np.square(np.log10(ypred + 1) - np.log10(yval + 1)).mean())
    print("=== evaluation metric = ", metric)
    return mlp


# def metric()


def train_keras(df_train_processed):
    xtrain, xval, ytrain, yval = train_test_split(
        df_train_processed.drop(columns=["Price"]),
        df_train_processed.Price,
        stratify=df_train_processed["PriceBinned"],
    )

    xtrain = xtrain.drop(columns=["PriceBinned"])
    xval = xval.drop(columns=["PriceBinned"])

    def create_model():
        model = Sequential()
        model.add(Dense(100, input_dim=len(xtrain.columns), activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1))
        adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            loss=tf.keras.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.AUTO
            ),
            optimizer=adam,
            metrics=tf.keras.metrics.MeanSquaredError(),
        )
        return model

    model = KerasClassifier(
        build_fn=create_model,
        epochs=1500,
        batch_size=100,
        verbose=1,
    )
    model.fit(xtrain, ytrain)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--trainer", choices=["mlp", "keras"], default="mlp")
    args = parser.parse_args()
    df_train_processed = process_df(df, target_file="df_train_processed.csv")
    df_test_processed = process_df(
        df_test, target_file="df_test_processed.csv", is_test=True
    )
    if args.trainer == "mlp":
        model = train_mlp_sklearn(df_processed=df_train_processed)
    elif args.trainer == "keras":
        model = train_keras(df_train_processed)
    test_prices = mlp.predict(df_test_processed)
    df_test["Price"] = test_prices
    df_test.to_excel("test_submission.xlsx")
