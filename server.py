from flask import Flask, request, jsonify
import utils
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'hi'

@app.route('/classify_image',methods=['GET','POST'])
def classify_image():
    image_data=request.form['image_data']
    response=jsonify(utils.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response





if __name__ == '__main__':
    utils.load_artifactes()
    app.run(debug=True,port=5000)