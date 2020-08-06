from flask import Flask
import os
from generateme.app import app
from generateme.app import index, showImageConv, showImageLinear, showImageStyle
IMAGE_FOLDER = os.path.join("static", "Photos")

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