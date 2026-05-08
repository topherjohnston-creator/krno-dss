FROM continuumio/miniconda3:latest

WORKDIR /app

# Install all dependencies via conda-forge — ecCodes is perfectly packaged here
RUN conda install -c conda-forge -y \
    eccodes \
    cfgrib \
    flask \
    flask-cors \
    requests \
    numpy \
    gunicorn \
    && conda clean -afy

COPY . .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
