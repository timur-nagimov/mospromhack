# SDK-free head/tail preflight

`python -m tpu.preflight` проверяет статический FP32 detect ONNX и сохраняет отдельные `head.onnx`, `tail.onnx`, `report.json`, `worker.log`. SDK, Docker, TPU и сеть не используются. Команды выполняются из корня checkout; проекту не нужна editable-установка.

Сначала можно проверить граф без ORT и без инференса:

```bash
python -m tpu.preflight \
  --onnx models/candidates/yolo11x-640/yolo11x-640.onnx \
  --graph-only --json
```

FP32-паритет выполняет полную сеть и цепочку head → tail на одних и тех же входах, последовательно на CPU. Запускать его следует отдельно от других тяжёлых заданий:

```bash
python -m tpu.preflight \
  --onnx models/candidates/yolo11x-640/yolo11x-640.onnx \
  --timeout 600 --json
```

По умолчанию сохраняется новый каталог `$GUARD_DATA_DIR/tpu_preflight/<UTC>-<id>` или `data/tpu_preflight/<UTC>-<id>` текущего checkout. `--output-dir` задаёт другой **ещё не существующий** каталог; прошлые результаты не перезаписываются. Таймаут по умолчанию 300 секунд, допустимо до 1800. После таймаута собственная process group получает TERM и KILL независимо от того, успел ли завершиться её лидер; на завершение отводится дополнительно до 5.2 секунды.

Обязательные входы паритета: нули, единицы и uniform [0,1) с seed 0. Дополнительно можно повторить `--image /absolute/path/frame.jpg` до 16 раз. Изображение преобразуется в RGB, letterbox с серым 114 и Pillow bilinear, затем NCHW float32 /255. Это вход для проверки эквивалентности разреза, не отдельная оценка точности preprocessing или детектора. JSON содержит SHA256 исходных изображений и фактических входных тензоров.

Граница выбирается по единственной поддерживаемой структуре в графе:

- DFL Conv с весами [0, …, reg_max−1] после Softmax и Sigmoid классов находятся в одном scope головы.
- DFL и классы происходят от разных выходов общей Split.
- DFL cut имеет форму [1,1,4,N], class cut — [1,C,N], исходный выход — [1,4+C,N].
- Обратный обход каждого исходного выхода должен завершаться на обеих cut tensors и константах. Остаточная зависимость от image input, отсутствующая/неиспользуемая граница или неподдерживаемый tail operator приводят к отказу до извлечения графов.

Префикс не закреплён за номером слоя. Поддерживаются канонические экспорты YOLOv8 с `/model.22` и YOLO11 с `/model.23`, если структура подтверждена; переименованная/изменённая структура требует отдельной поддержки. Две подходящие головы считаются неоднозначностью. `--cut DFL_TENSOR CLASS_TENSOR` позволяет явно проверить ожидаемые имена, но не обойти структурную проверку. Выданные report.head.cut можно передать явному `--cut` существующего `tpu/compile.py` в отдельном SDK-процессе. У compile.py это одна строка через запятую, например `--cut '/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0'`, а не два аргумента. Логика SDK не изменена; defaults его путей теперь вычисляются из текущего checkout, `GUARD_ROOT` и `GUARD_MODELS_DIR`.

Не поддерживаются dynamic/unknown/zero dimensions, batch >1, FP16, внешние tensor data, custom ops/functions, control flow, stochastic ops, pose/segmentation/NMS. Вход ограничен 1×3×H×W, не больше 3×2048×2048 элементов. Проверяются ONNX checker, inference форм, конечность float-весов/констант и полученных FP32 тензоров. Извлечение использует [ONNX Extractor](https://onnx.ai/onnx/api/utils.html); ORT настроен на [CPU sessions](https://onnxruntime.ai/docs/api/python/api_summary.html) с intra/inter threads 2/2, последовательным исполнением и отключённой telemetry.

`report.json` содержит формы на границе, opsets, operator inventory для full/head/tail, SHA256 ONNX/runner, версии библиотек, SHA256 тестовых входов и per-output max/mean absolute difference, число элементов и число нарушений допуска. Условие: `abs(full − split) <= atol + rtol * abs(full)` для каждого элемента; defaults atol=rtol=1e-4. Неверная форма, dtype или NaN/Inf — ошибка. Допуски записываются в отчёт, их изменение не является свидетельством улучшения модели.

Статусы:

| status | Что доказано |
|---|---|
| graph_validated | Граница найдена, зависимости закрыты, head/tail извлечены; FP32 inference пропущен |
| fp32_parity_passed | Дополнительно пройден паритет всех объявленных CPU-входов |
| error / timeout / interrupted | Preflight не завершён успешно; подробности и доступные промежуточные артефакты сохранены |

Exit codes: 0 — выбранные проверки пройдены; 1 — ошибка/таймаут; 2 — неверный CLI или существующий output directory; 130/143 — прерывание SIGINT/SIGTERM. `--json` выводит один JSON в stdout; диагностика worker находится в файле. Собственная worker group очищается также после обычного завершения, исключения или прерывания родителя. Итоговый interrupted/error report записывается после остановки worker, чтобы поздний результат не заменил его на success.

Ни один статус не означает compiled. Поля `tpu_compilation=not_attempted`, `tpu_compatibility=unknown`, `quantized_parity=not_checked` отделяют результаты CPU от последующих стадий. Operator inventory показывает, в частности, MatMul/Softmax, но не доказывает поддержку SDK. Для TPU всё ещё нужны компиляция, калибровка/квантование, проверка quantized/full результатов и реальный device/backend прогон.

В этой фиче FP32-паритет проверен только на маленьких детерминированных ONNX-графах с реальными Conv/Softmax/Sigmoid, включая намеренно повреждённый хвост. У локального YOLO11x-640 прочитан и проверен граф: 618 узлов, голова /model.23, 4 MatMul и 3 Softmax; tail зависит от двух cut tensors без исходного image input. Инференс этой модели, SDK-компиляция и TPU-проверка в рамках фичи не выполнялись.
