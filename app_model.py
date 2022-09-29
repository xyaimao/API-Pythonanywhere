from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido al API del modelo Advertising de Xin"


# mirar todo el base de datos:

@app.route('/all', methods=['GET'])
def get_all():
    connection = sqlite3.connect('Advertising.db')
    cursor = connection.cursor()
    select_datos = "SELECT * FROM Advertising"
    result = cursor.execute(select_datos).fetchall() 
    connection.close()
    return jsonify(result)


# Ofrezca la predicción de ventas a partir de todos los valores de gastos en publicidad. (/predict)
@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open('advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'


# Un endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada. (/ingest_data)

@app.route('/ingest_data', methods = ['POST'])

def nuevo_registro():
    
    tv = float(request.args["TV"])
    radio = float(request.args["radio"])
    newspaper = float(request.args["newspaper"])
    sales = float(request.args["sales"])

    connection = sqlite3.connect('data/Advertising.db')
    cursor = connection.cursor()
    insert_data = "INSERT INTO Advertising VALUES (?,?,?,?)"
    result = cursor.execute(insert_data, (tv,radio,newspaper,sales)).fetchall()
    connection.commit() 
    connection.close()
    return ("se ha añadido nuevos datos: " + str(tv) + " " + str(radio)+ " "+ str(newspaper)+ " " + str(sales))

# Posibilidad de reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan. (/retrain)

@app.route('/retrain', methods=['PUT'])
def retrain():

    connection = sqlite3.connect('Advertising.db')
    cursor = connection.cursor()
    select_data = "SELECT * FROM Advertising"
    result = cursor.execute(select_data).fetchall()
    names = [description[0] for description in cursor.description]

    df = pd.DataFrame(result,columns=names)

    connection.close()
    
    X = df.drop(columns=['sales'])
    y = df['sales']

    model = pickle.load(open('advertising_model','rb'))
    model.fit(X,y)
    pickle.dump(model, open('advertising_model_v1','wb'))

    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

    return "New model retrained and saved as advertising_model_v1. The results of MAE with cross validation of 10 folds is: " + str(abs(round(scores.mean(),2)))




#app.run()