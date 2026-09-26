FROM python:3.12-slim

WORKDIR /app

# Installed before copying app code so this layer is cached across rebuilds
# that only change api/main.py, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what serving actually needs: the API code and the exported model.
# No src/ (training-only), no data/, no mlruns/ -- keeps the image small
# and means a broken training script can never break the deployed API.
RUN useradd -m -u 1000 user

COPY --chown=user:user api/ api/
COPY --chown=user:user artifacts/ artifacts/

USER user

EXPOSE 7860
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}