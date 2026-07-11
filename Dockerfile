FROM python:3.11-slim

# system deps needed by cvxpy solvers (ECOS/SCS/Clarabel) + numpy/scipy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first so Docker layer caches when only source changes
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copy engine + api + data + examples
COPY engine ./engine
COPY api ./api
COPY cli.py ./cli.py
COPY data ./data
COPY examples ./examples

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# long timeout because min_max_drawdown DE can take ~40s
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", \
     "--workers", "2"]
