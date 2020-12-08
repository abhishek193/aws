FROM python:3.6

# set the working directory
RUN ["mkdir", "app"]
WORKDIR "app"

# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

COPY "BatchCreator.ipynb" .
#COPY BatchCreator.py /app/BatchCreator.py
COPY "Train.ipynb" /app/Train.ipynb

# install environment dependencies
ENV SHELL /bin/bash

# provision environment
#EXPOSE 8080

# VOLUME ["/home/tarun/Downloads" "/tmp"]
# If the following dependency is put in requirements, it gets ignored as 1.0.8 is already found but that version poses issues in code
RUN ["pip", "install", "--upgrade", "Keras-Applications==1.0.7"]
COPY BatchCreator.py /app/BatchCreator.py
COPY "Train.py" /app/Train.py

COPY "run.sh" .
RUN ["chmod", "+x", "./run.sh"]
COPY training-data/ /tmp/training-data/
RUN ["mkdir", "-p" , "/app/training-data"]
COPY training-data/* /app/training-data/

COPY "preparemodel.py" /app/preparemodel.py
RUN ["pip", "install", "boto3"]
ENTRYPOINT ["./run.sh"]
CMD ["train"]
