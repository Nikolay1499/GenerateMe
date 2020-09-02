#Use this file when we want to deploy. Flask has a built-in development server 
#while gevent can be used as a production WSGI server
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
from generateme.app import app

http_server = WSGIServer(("", 5000), app)
http_server.serve_forever()