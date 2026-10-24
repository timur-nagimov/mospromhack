#!/usr/bin/env bash
# Установка LinQ tpu_framework 2026.9.45 на Jupyter-сервер стенда (Linux x86_64, conda Python 3.11).
# SDK требует Python 3.12 (tpu_framework.pdf стр. 5) -> отдельное окружение ~/linq312 через uv.
# Колёса SDK загружаются заранее: python tools/jhub.py upload --parallel 4 <whl> linq_sdk/packets/<whl>
# (с сервера cloud.lin-q.ru недоступен, PyPI и download.pytorch.org — доступны).
# Запуск: python tools/jhub.py run-bg --name sdk_env 'bash ~/work/tpu/server_setup.sh'
set -euo pipefail
PK=${PK:-$HOME/work/linq_sdk/packets}
ENV=${ENV:-$HOME/linq312}
python3 -m pip install --user -q uv
UV=$HOME/.local/bin/uv
[ -x "$ENV/bin/python" ] || "$UV" venv -q -p 3.12 "$ENV"
# torch — CPU-сборка (2.9.1+cpu удовлетворяет пину ==2.9.1 из dnn_quant), иначе тянется CUDA
cd "$PK"  # requirements.txt ссылается на колёса относительными путями
"$UV" pip install -p "$ENV/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  --find-links "$PK" -r "$PK/requirements.txt" pillow pika protobuf
"$ENV/bin/python" - <<'EOF'
import time; t = time.time()
import tpu_framework, torch, tensorflow, onnx, numpy
from tpu_framework import O1, O5, TPU_H1M
print("SDK_OK import %.1fs" % (time.time() - t), "torch", torch.__version__, "tf", tensorflow.__version__,
      "onnx", onnx.__version__, "numpy", numpy.__version__)
EOF
