from flask import Flask, render_template, send_file, Response, url_for
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
app.config["TEMPLATES_AUTO_RELOAD"] = True

def create_app():
  app = Flask(__name__)
  app.config["UPLOAD_FOLDER"] = IMAGE_FOLDER
  app.config["TEMPLATES_AUTO_RELOAD"] = True
  app.add_url_rule("/", "index", index)
  app.add_url_rule("/index", "index", index)
  app.add_url_rule("/getStyleImage", "showImageStyle", showImageStyle)
  app.add_url_rule("/getConvImage", "showImageConv", showImageConv)
  app.add_url_rule("/getLinearImage", "showImageLinear", showImageLinear)
  return app

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/getStyleImage")
def showImageStyle():
    getStyleImage()
    return getImage()
    
@app.route("/getConvImage")
def showImageConv():
    getConvImage()
    return getImage()
    
@app.route("/getLinearImage")
def showImageLinear():
    getLinearImage()
    return getImage()

def getImage():
    img = Image.open("static/Photos/image.png")
    file_object = io.BytesIO()

    img.save(file_object, "PNG")  
    file_object.seek(0)

    response = send_file(file_object, mimetype="image/PNG")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response