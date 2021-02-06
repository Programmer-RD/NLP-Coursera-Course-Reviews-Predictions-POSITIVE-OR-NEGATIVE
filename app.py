from flask import *
from sklearn.feature_extraction.text import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import json
import pickle
from flask_restful import *
from flask_restful import reqparse

app = Flask(__name__)
app.debug = True
app.secret_key = "Ranuga D 2008"
api = Api(app)
data = reqparse.RequestParser()
data.add_argument(name="Review", required=True, type=str, help="The review")


def predict(review):
    model = pickle.load(open("./model.pkl", "rb"))
    cv = pickle.load(open("./cv.pkl", "rb"))
    text = review
    text = cv.transform([text])
    info_dict = open("./info.json", "r")
    info_dict = json.load(info_dict)
    result = model.predict(text)[0]
    result_ = result
    result = info_dict[f"{result}"]
    info_dict_2 = {}
    for keys, values in zip(info_dict.keys(), info_dict.values()):
        info_dict_2[values] = keys
    result_ = f"{result_}"
    return [result, info_dict, result_, info_dict_2,('Over 2.5 Gets POSITIVE and NEGATIVE get under 2.5')]


class Predict(Resource):
    def get(self):
        info = data.parse_args()
        result = predict(info["Review"])
        return {"result": result}

    def post(self):
        info = data.parse_args()
        result = predict(info["Review"])
        print(result)
        return {"result": result}


# api.add_resource(Predict,'/api')
api.add_resource(Predict, "/api/")

if __name__ == "__main__":
    app.run()
