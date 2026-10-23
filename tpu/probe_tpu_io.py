"""Исследование упаковщика tpu_io (SDK) для raw-режима: сигнатуры и сверка с numpy-упаковкой. Запуск в ~/linq312."""
import inspect
import json
import sys
import zipfile

import numpy as np
import tpu_io

for n in ["pack", "unpack", "tpu_encode", "tpu_decode", "get_tpu_shape", "tensor_to_words", "words_to_tensor",
          "pack_mapping", "unpack_mapping", "get_padding", "requantize"]:
    f = getattr(tpu_io, n)
    try:
        sig = str(inspect.signature(f))
    except (TypeError, ValueError):
        sig = "(?)"
    print(f"## {n}{sig}\n{(f.__doc__ or '').strip()[:600]}\n")

if len(sys.argv) > 1:
    prog = sys.argv[1]
    meta = json.loads(zipfile.ZipFile(prog).read("metadata.json"))
    print("hw ddr_word_len", meta["hardware_parameters"]["ddr_word_len"])
    for sec in ("inputs", "outputs"):
        for rid, d in meta[sec].items():
            for name, t in d.items():
                print(sec, rid, name, {k: v for k, v in t.items() if k not in ("scales", "zero_points")},
                      "scales[:3]", t["scales"][:3], "n_scales", len(t["scales"]))
