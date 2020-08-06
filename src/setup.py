from setuptools import setup

setup(name = "generateme",
      version = "0.1",
      description = "Flask application to generate images with Generative Adversarial networks",
      url = "https://github.com/Nikolay1499/GenerateMe",
      author = "Nikolay Valkov",
      author_email = "nikolay1499@gmail.com",
      license = "MIT",
      include_package_data = True,
      packages = ["generateme"],
      install_requires = [
          "flask",
          "gevent",
          "numpy",
          "Pillow",
          "matplotlib",
          "future",
      ],
      zip_safe = False
)