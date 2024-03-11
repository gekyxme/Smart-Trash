from features.scripts import model, predict_image, predict_external_image
from flask import Flask, render_template as template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)
    predicted_class = predict_external_image(imagepath, model)
    return template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(port=3000, debug=True)