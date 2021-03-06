from sqlalchemy import create_engine
import pandas as pd
import argparse
import numpy as np
import json
import pickle
from pathlib import Path

from utils import tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from typing import List, Tuple, Dict, Set


def load_data(database_filepath: str) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Load data from database

    :param database_filepath: path to sqlite database file
    :return: tuple of feature matrix, target matrix and list of category names
    """
    repo = Path.cwd()
    path = repo / database_filepath

    engine = create_engine(f"sqlite:///{path}")
    df = pd.read_sql("select * from Message", con=engine)

    X = df["message"].values
    Y = df.iloc[:, :-3].values
    output_labels = list(df.iloc[:, :-3].columns)

    return X, Y, output_labels


def build_model() -> Pipeline:
    """
    Build pipeline and run grid search with cross-validation

    :return: Model object returned by GridSearchCV
    """

    pipeline = Pipeline(
        [
            ("cv", CountVectorizer(tokenizer=tokenize)),
            ("transformer", TfidfTransformer()),
            (
                "rfc",
                MultiOutputClassifier(
                    LogisticRegression(
                        penalty="l1",
                        C=0.10,
                        multi_class="ovr",
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


def get_metrics(y_test: np.ndarray, y_preds: np.ndarray) -> Dict:
    """
    Generate classification metrics for a single category

    :param y_test: Vector of target values
    :param y_preds: Vector of predictions from classifier
    :return: Dictionary with classification metrics
    """
    cr = classification_report(y_test, y_preds, zero_division=0, output_dict=True)
    return cr


def generate_report(Y_test: np.ndarray, Y_preds: np.ndarray, category_names) -> Dict:
    """
    Generate classification metrics for all categories

    :param Y_test: Matrix of target values
    :param Y_preds: Matrix of predictions
    :param category_names: Names of categories
    :return: Dictionary with classification metrics for all categories
    """
    report_dict = {}
    for idx, output in enumerate(category_names):
        metrics_dict = get_metrics(Y_test[:, idx], Y_preds[:, idx])
        report_dict[output] = metrics_dict

    return report_dict


def evaluate_model(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    category_names: List,
    report: bool = False,
) -> None:
    """
    Evaluate model returned by GridSearchCV. Generate classification metrics.
    Save metrics to file and print them if requested

    :param model: Model returned by GridSearchCV
    :param X_test: Feature matrix
    :param Y_test: Target matrix
    :param category_names: Names of categories
    :param report: Enable printing of classification metrics
    :return: None
    """
    Y_preds = model.predict(X_test)
    report_dict = generate_report(Y_test, Y_preds, category_names)

    with open("./models/model_results.json", "w", encoding="utf-8") as fhand:
        json.dump(report_dict, fhand)

    if report:
        for k, v in report_dict.items():
            print(
                f"""category: {k} \n accuracy: {v['accuracy']:.4f}, precision: {v['1']['precision']:.4f}, recall: {v['1']['recall']:.4f}, f-score: {v['1']['f1-score']:.4f}"""
            )


def save_model(model, model_filepath: str) -> None:
    """
    Save model returned by GridSearchCV

    :param model: Model returned by GridSearchCV
    :param model_filepath: Path of pickle file
    :return: None
    """

    with open(model_filepath, "wb") as fhand:
        pickle.dump(model, fhand)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("Database", help="filepath: database file")
    parser.add_argument("Model", help="filepath: model pickle file")
    args = parser.parse_args()

    X, Y, category_names = load_data(args.Database)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, Y_test, category_names)

    print("Saving model...\n    MODEL: {}".format(args.Model))
    save_model(model, args.Model)

    print("Trained model saved!")


if __name__ == "__main__":
    main()
