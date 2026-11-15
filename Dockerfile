FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir piper-tts
COPY live-call-server.py live-call-page.html receptionist-system-prompt.txt ./
RUN mkdir -p models \
 && curl -L -o models/fa_IR-amir-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx" \
 && curl -L -o models/fa_IR-amir-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx.json"
ENV PIPER_MODEL=/app/models/fa_IR-amir-medium.onnx
ENV HOST=0.0.0.0
# PORT is injected by Render; the server reads it (default 8766 locally).
CMD ["python", "live-call-server.py"]
