from flask import Flask, render_template, send_file, Response, url_for, jsonify
from PIL import Image
import numpy as np
import io
import os
import pybase64
from generateme.LinearGan import getLinearImage
from generateme.DCGan import getConvImage
from generateme.StyleGan import getStyleImage

IMAGE_FOLDER = os.path.join("static", "Photos")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = IMAGE_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True
folder = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/getStyleImage")
def showImageStyle():
    return getImage(getStyleImage())
    
@app.route("/getConvImage")
def showImageConv():
    return getImage(getConvImage())
    
@app.route("/getLinearImage")
def showImageLinear():
    return getImage(getLinearImage())
    
@app.route("/api/getStyleImage")
def showImageStyleApi():
    return getJson(getStyleImage())
    
@app.route("/api/getConvImage")
def showImageConvApi():
    return getJson(getConvImage())
    
@app.route("/api/getLinearImage")
def showImageLinearApi():
    return getJson(getLinearImage())

def getImage(img):
    file_object = io.BytesIO()
    img.save(file_object, "PNG")  
    file_object.seek(0)

    response = send_file(file_object, mimetype="image/PNG")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response
    
def getJson(img):
    file_object = io.BytesIO()
    img.save(file_object, "PNG")  
    file_object.seek(0)
    base64image = (pybase64.b64encode(file_object.getvalue(), altchars="_:")).decode("utf-8")
    response =  { "Completion_Status": "Success", "Image": base64image}
    return jsonify(response) # send the result to client
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000)