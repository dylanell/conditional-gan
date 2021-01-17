# Install base image optimized for Python.
FROM python:3.8-slim-buster

# pytorch base image
# FROM pytorch/pytorch

# add requirements file to base image
COPY requirements.txt /requirements.txt

# install python packages to base image
RUN pip install -r requirements.txt

# copy source code in current directory to the container
ADD . /app

# set working directory to directory with container executable
WORKDIR /app/server

# expose the port the app runs on
EXPOSE 8080

# command to run api
CMD ["python", "api.py"]

# create bash entry point for debugging
# run $ docker exec -it cgan /bin/bash
# ENTRYPOINT ["tail", "-f", "/dev/null"]
