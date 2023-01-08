FROM python:3.9

RUN mkdir -p /usr/src/app/

WORKDIR /usr/src/app/

ENV STATIC_URL /static

COPY . /usr/src/app/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python" ]

CMD ["model_web_app.py" ]

