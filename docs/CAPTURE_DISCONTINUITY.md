# Граница capture_id при loop/reconnect

Непустой новый `capture_id` отделяет сессии одной камеры перед анализом её следующего кадра. Первый известный ID — bootstrap; тот же ID и пустой legacy ID состояние не сбрасывают. ID камеры, process/run ID, source sequence, счётчик обработанных кадров и накопленные метрики сохраняются. Detector/backend, модели и пороги не меняются.

Порядок границы:

1. `EventManager.capture_closures(camera_id)` возвращает deep snapshots открытых событий с `ended_at=last_violation_ts` каждого события и фактической накопленной active duration. Метод не меняет manager: нет искусственного пустого кадра, нового wall timestamp или очистки до записи.
2. Worker удерживает замороженный список закрытий. `PipelineEngine.close_capture` сохраняет и читает обратно каждое закрытие перед event_update. Уже подтверждённые элементы убираются из pending списка, поэтому ошибка следующей записи не повторяет прежнюю notification/revision. При неопределённом результате commit сначала проверяется journal, без повторной outbox revision. Edge использует существующий persistence retry/outbox.
3. Только после подтверждения всех закрытий выполняется `reset(camera_id)` и создаются новые person/object Tracker и экземпляры действующих правил с прежними параметрами. Кандидаты, aliases, cooldown, атрибуты, motion/rule history, старый overlay и JPEG caches не переходят в новую сессию.

Обновление journal частичное: ended_at и machine details active_duration_s/reopened_count. Confirmation bbox, исходные start/confirm/origin, snapshots, clip path, операторские status/note сохраняются. Старые события и outbox/receipts из SQLite не удаляются. Числовой track_id нового трекера может снова начинаться с 1; он относится к новому capture_id. Новый инцидент получает отдельный event ID и проходит обычное окно подтверждения.

При ошибке закрытия новый кадр не попадает в detector/tracker/rules/manager. `status().capture_boundary_pending` и retryable capture error показывают причину. Следующий кадр повторяет pending boundary; если за это время источник уже сменил B на C, после подтверждения старой A принимается фактический C, без выдуманного события B. Stop ждёт завершения analysis thread и пытается дописать pending closures. При join/write failure worker, journal и backend остаются доступны для повторного stop после восстановления. Обычная shutdown policy вне capture boundary остаётся прежней.

Source callback отделяет media раньше worker:

- Ring атомарно rotates/seals по source capture_id. Старые queued/active jobs держат свой старый ring; новые кадры туда не добавляются.
- Для позднего события выбирается только ring его origin capture. Lookup хранит current и один previous ring на камеру; если A уже вытеснен при A→B→C, schedule возвращает False и штатный caller отмечает clip_error, вместо подстановки B/C.
- Sealed ring завершается на реально имеющихся кадрах. Постролл может быть короче заявленного окна; новая сцена его не дополняет. При недостатке кадров действует обычная ошибка clip writer.
- Число jobs/writers и frame caps сохраняются. Общий budget делится также между живыми sealed rings, удерживаемыми заданиями. Source использует порядок media lock → ring lock и не обращается к journal/worker persistence. Writer callbacks вызываются без этих locks.
- Live/analyzed overlay не накладывает старые boxes на raw packet с другим непустым capture_id, даже если новый кадр ещё не обработан.

Legacy источники без capture_id сохраняют прежнюю непрерывность: без identity граница не может быть установлена автоматически. При ручной сборке FramePacket для одной сессии передавайте один постоянный capture_id (или явно `""` для legacy); default FramePacket создаёт новый UUID для отдельного пакета. Реальные источники уже передают свой постоянный ID и меняют его на loop/reconnect.

API двухфазный и пригоден для offline replay: `capture_closures → persist/readback → reset`. Pending snapshots до commit удерживаются в памяти worker; новый протокол восстановления незавершённого manager state после аварийного завершения процесса эта фича не добавляет. У штатного SQLite Journal успешная запись и outbox атомарны; совместимость MemoryJournal означает readback в памяти, а не сохранность на диске.

Проверка без моделей/сети:

```bash
python -m pytest -q \
  tests/test_capture_discontinuity.py tests/test_events_manager.py \
  tests/test_media.py tests/test_pipeline_core.py tests/test_pipeline_e2e.py \
  tests/test_pipeline_gateway.py tests/test_edge_gateway.py
```

Регрессии проверяют same-position/new-capture, pending candidate, невозможность reopen старого event ID, изоляцию двух камер, reset атрибутов/правил, last violation и active duration после reopen, before/uncertain-after-commit, partial-batch ack/outbox/notification, stop/join recovery, source callback во время pending boundary, overlay и старые active/queued clips при нескольких быстрых сменах capture. Настоящие ONNX/TPU inference и RTSP в этих проверках не запускаются.
