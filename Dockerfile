FROM gw000/keras:2.1.4-py3


COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt