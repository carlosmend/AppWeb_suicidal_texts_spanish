from flask import Flask
from flask_cors import CORS
from flask import Flask, jsonify, request,render_template
from models.predict import predict
from models.clean import cleanup


app=Flask(__name__,template_folder='templates')
CORS(app, origins="*")


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predecir():
    print(request)
    text = request.json['texto']   
    predictions,moda=predict(cleanup(text))
    return jsonify({'predictions':predictions,'moda':moda})

if __name__ == '__main__':
        app.run(debug=True, port=5000)