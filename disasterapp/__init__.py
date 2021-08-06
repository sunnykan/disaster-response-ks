from flask import Flask

app = Flask(__name__)

from disasterapp.utils import tokenize
from models.utils import tokenize
from disasterapp import routes
