FROM python:3.7
FROM pytorch/pytorch

MAINTAINER Nikolay Valkov nikolay1499@gmail.com

# set a directory for the app
WORKDIR /usr/app/

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/app/src

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./pywsgi.py"]
