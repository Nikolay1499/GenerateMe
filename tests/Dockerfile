FROM python:3.7
FROM pytorch/pytorch

MAINTAINER Nikolay Valkov nikolay1499@gmail.com

# set a directory for the app
WORKDIR /usr/app/

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest==6.0

WORKDIR /usr/app/src/

# install dependencies
RUN pip install -e .

WORKDIR /usr/app/tests

# run the command
CMD ["pytest"]
