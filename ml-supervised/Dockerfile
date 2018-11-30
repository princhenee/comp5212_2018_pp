FROM ubuntu

RUN apt-get update
RUN apt-get install -y python-tkinter python-pip
COPY * /srv
WORKDIR /srv
RUN pip install -r requirements.txt
