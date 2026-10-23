#!/usr/bin/env bash
# Окружение конвейера guard на Jupyter-стенде (D-005): Python 3.11 venv ~/guard311 с зависимостями pyproject.toml.
# Код заливается tpu/sync_to_jupyter.py в ~/work/guard_repo.
# Запуск: python tools/jhub.py run-bg --name guard_env 'bash $HOME/work/guard_repo/tpu/server_guard_env.sh'
# Контейнер стенда: 1.5 CPU, 4 ГиБ RAM -> ставим OMP_NUM_THREADS=2 при запуске конвейера.
set -euo pipefail
ENV=${ENV:-$HOME/guard311}
python3 -m pip install --user -q uv
UV=$HOME/.local/bin/uv
[ -x "$ENV/bin/python" ] || "$UV" venv -q -p 3.11 "$ENV"
"$UV" pip install -p "$ENV/bin/python" numpy "opencv-python-headless<4.13" onnxruntime onnx "fastapi>=0.110" \
  "uvicorn[standard]" "pydantic>=2.6" pyyaml pika aio-pika scipy lap imageio-ffmpeg python-multipart requests \
  websocket-client pillow protobuf pytest pytest-asyncio httpx
"$ENV/bin/python" - <<'EOF'
import numpy, cv2, onnxruntime, pika, scipy, lap, pydantic, yaml, fastapi
print("GUARD_ENV_OK numpy", numpy.__version__, "cv2", cv2.__version__, "ort", onnxruntime.__version__,
      "pydantic", pydantic.__version__)
EOF
cd $HOME/work/guard_repo 2>/dev/null && "$ENV/bin/python" -c "import guard.inference.linq_tpu, guard.inference.onnx_cpu; print('GUARD_IMPORT_OK')" || true
