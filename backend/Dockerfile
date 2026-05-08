FROM mambaorg/micromamba:1.5-jammy

USER root
WORKDIR /app

# Install Python + ecCodes + all dependencies via conda-forge
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    eccodes \
    cfgrib \
    flask \
    flask-cors \
    requests \
    numpy \
    gunicorn \
    && micromamba clean --all --yes

COPY . .

# Make conda environment available
ENV PATH="/opt/conda/bin:$PATH"

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
