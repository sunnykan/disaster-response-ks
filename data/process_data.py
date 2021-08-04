import argparse
import pandas as pd

from pathlib import Path
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load data sets for messages and categories; and merge them

    :param messages_filepath: path to messages data file
    :param categories_filepath: path to categories data file
    :return: dataframe containing categories and messages
    """
    repo = Path.cwd()

    path = repo / messages_filepath
    messages = pd.read_csv(path, sep=",")

    path = repo / categories_filepath
    categories = pd.read_csv(path, sep=",")

    messages.drop_duplicates(subset="id", inplace=True)
    categories.drop_duplicates(subset="id", inplace=True)

    messages.set_index(keys="id", inplace=True)
    categories.set_index(keys="id", inplace=True)

    df = pd.merge(categories, messages, left_index=True, right_index=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data

    :param df: dataframe with categories and messages
    :return: cleaned data frame
    """
    categories = df.categories.str.split(pat=";", expand=True)
    category_colnames = list(map(lambda x: x.split("-")[0], categories.iloc[0, :]))
    categories.columns = category_colnames
    categories = categories.applymap(lambda x: int(x[-1]))
    categories.drop(columns="child_alone", inplace=True)

    df.drop(columns="categories", inplace=True)
    df = pd.concat([categories, df], join="inner", axis=1)
    df.drop_duplicates(inplace=True)
    df = df[df.related != 2]

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    Save dataframe to sqlite database

    :param df: dataframe to be saved to database
    :param database_filename: filename of sqlite database
    :return: None
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("Message", engine, if_exists="replace", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Messages", help="filepath: messages data")
    parser.add_argument("Categories", help="filepath: categories data")
    parser.add_argument("Database", help="filepath: Database")
    args = parser.parse_args()

    merged_df = load_data(args.Messages, args.Categories)
    cleaned_df = clean_data(merged_df)
    save_data(cleaned_df, args.Database)


if __name__ == "__main__":
    main()
