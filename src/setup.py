from setuptools import setup

def readme():
    with open("README.rst") as f:
        return f.read()

setup(name = "generateme",
      version = "0.1.2",
      description = "Flask application to generate images with Generative Adversarial networks",
      long_description = readme(),
      url = "https://github.com/Nikolay1499/GenerateMe",
      author = "Nikolay Valkov",
      author_email = "nikolay1499@gmail.com",
      license = "MIT",
      packages = ["generateme"],
      install_requires = [
          "flask",
          "gevent",
          "numpy",
          "Pillow",
          "matplotlib",
          "future",
          "pybase64",
      ],
      zip_safe = False,
      include_package_data = True,
)