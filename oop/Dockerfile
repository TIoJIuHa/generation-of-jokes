# docker run --rm --name web -p 5000:5000 -v (absolute path to folder "static"):/usr/src/app/static -v (absolute path to folder "templates"):/usr/src/app/templates webapp.1

FROM python:3.9

RUN mkdir -p /usr/src/app/

WORKDIR /usr/src/app/

ENV STATIC_URL /static

COPY . /usr/src/app/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python" ]

CMD ["model_web_app.py" ]

