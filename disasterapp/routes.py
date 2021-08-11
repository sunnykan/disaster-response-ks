from disasterapp import app

import json
import plotly
import pandas as pd
import joblib

from flask import render_template, request, jsonify
import plotly.graph_objects as gro
from sqlalchemy import create_engine
from utility.utils import tokenize
from typing import List


# load data
engine = create_engine("sqlite:///./data/DisasterResponse.db")
df = pd.read_sql_table("Message", engine)

# load model
with open("./models/model_logistic.pkl", "rb") as fhand:
    model = joblib.load(fhand)

# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    figures = []

    # FIGURE 1 ------------------------------
    prop_messages = (df.iloc[:, :-3].mean().sort_values(ascending=False) * 100)[:10]

    figure = gro.Figure(
        [gro.Bar(name="Categories", x=prop_messages.index, y=prop_messages.values)]
    )

    figure.update_layout(
        title="Distribution of Message Categories (Top 10)",
        yaxis_title="Proportion of Messages by Category",
    )
    figures.append(figure)

    # FIGURE 2 ------------------------------
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    figure = gro.Figure([gro.Bar(name="Genres", x=genre_names, y=genre_counts)])

    figure.update_layout(
        title="Distribution of Message Genres", yaxis_title="Number of Messages",
    )
    figures.append(figure)

    # FIGURE 3 ------------------------------
    aid_genre = df.groupby(["genre"])["aid_related"].sum()
    weather_genre = df.groupby(["genre"])["weather_related"].sum()

    figure = gro.Figure(
        [
            gro.Bar(name="Aid", x=genre_names, y=aid_genre),
            gro.Bar(name="Weather", x=genre_names, y=weather_genre),
        ]
    )
    figure.update_layout(
        barmode="group",
        title="Aid and Weather related Messages by Genre",
        yaxis_title="Number of Messages",
    )
    figures.append(figure)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[:-3], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )
