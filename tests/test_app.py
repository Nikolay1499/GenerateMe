import os
import pytest
from flask import Flask
from generateme import create_app


@pytest.fixture
def app():
    app = create_app()
    yield app

@pytest.fixture
def client(app):
    return app.test_client()
    
def test_static(client, app):
    assert client.get("/static/Photos/icon.png").status_code == 200

def test_index(client, app):
    print(app.url_map)
    assert client.get("/").status_code == 200
    assert client.get("/index").status_code == 200
  
def test_showImageStyle(client, app):
    assert client.get("/getStyleImage").status_code == 200
  
def test_showImageConv(client, app):
    assert client.get("/getConvImage").status_code == 200
 
def test_showImageLinear(client, app):
    assert client.get("/getLinearImage").status_code == 200
