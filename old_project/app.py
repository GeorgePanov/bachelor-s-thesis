import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("develop a model/heart.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=['GET', 'POST'])
def predict():
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['chest_pain_type']
    data4 = request.form['resting_bp']
    data5 = request.form['cholesterol']
    data6 = request.form['fasting_bs']
    data7 = request.form['resting_ecg']
    data8 = request.form['max_hr']
    data9 = request.form['exercise_angina']
    data10 = request.form['oldpeak']
    data11 = request.form['st_slope']

    arr = np.array([[data1, data2, data3, data4, data5, data6,
                   data7, data8, data9, data10, data11]])
    pred = model.predict(arr)
    pred_proba = model.predict_proba(arr)

    try:
        if pred == 1:
            return render_template("index.html", prediction_text="Срочно идите к кардиологу", predict_probability=pred_proba[0][1].round(4))
        elif pred == 0:
            return render_template("index.html", prediction_text="Вы можете не торопиться идти к кардеологу", predict_probability=pred_proba[0][1].round(4))
    except:
        return render_template("index.html", prediction_text="Что-то сломалось")


if __name__ == "__main__":
    flask_app.run(debug=True)
