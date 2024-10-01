FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc

RUN pip install -U pip
RUN pip install poetry==1.5.1

WORKDIR /app

COPY ["pyproject.toml", "config.yaml", "README.md","./"]
ADD src src/

RUN poetry build
RUN pip install dist/airqualityforecast-0.1.0-py3-none-any.whl

ENTRYPOINT [ "inference_pipeline", "config.yaml" ]
