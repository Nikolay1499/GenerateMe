from flask import Flask, render_template, send_file, Response
from PIL import Image
import numpy as np
import io
import os
from LinearGan import getLinearImage
from DCGan import getConvImage
from StyleGan import getStyleImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random

IMAGE_FOLDER = os.path.join("static", "Photos")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = IMAGE_FOLDER


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/getStyleImage")
def saveImageStyle():
    return getImage(getStyleImage())
    
@app.route("/getConvImage")
def saveImageConv():
    return getImage(getConvImage())
    
@app.route("/getLinearImage")
def saveImageLinear():
    return getImage(getLinearImage())

def getImage(arr):
    img = Image.fromarray(arr.astype('uint8'))

    file_object = io.BytesIO()

    img.save(file_object, 'png')  
    file_object.seek(0)

    return send_file(file_object, mimetype='image/png')