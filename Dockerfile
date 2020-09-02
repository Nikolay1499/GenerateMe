FROM python:3.7
FROM pytorch/pytorch

MAINTAINER Nikolay Valkov nikolay1499@gmail.com

# set a directory for the app
WORKDIR /usr/app/

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/app/src/

RUN pip install -e .

WORKDIR /usr/app

# tell the port number the container should expose
EXPOSE 5000

ENV FLASK_APP=generateme

# run the command
CMD ["flask", "run", "--host=0.0.0.0"]