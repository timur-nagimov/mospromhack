# Настоящая RabbitMQ: opt-in acceptance

Скрипт запускает отдельные процессы edge и gateway с настоящими `RabbitPublisher`/`RabbitConsumer`, двумя SQLite и HTTP/WebSocket API. Источник события синтетический, снапшот — настоящий JPEG. Модели и камеры не запускаются; проверка CPU pipeline выполняется отдельно. `.env` не читается. AMQP URL принимается только из `GUARD_RABBIT_URL`, не попадает в аргументы процессов, логи и отчёт.

Нужны Python окружение проекта, зависимости проекта (включая pika, uvicorn, websocket-client, Pillow), установленные RabbitMQ и Erlang. На macOS можно установить `brew install rabbitmq`; системный сервис запускать не требуется.

Из корня репозитория поднимите собственный loopback узел на свободных портах:

```sh
.venv/bin/python -m tools.local_rabbit start \
  --state-dir "$PWD/data/rabbit-local" \
  --port 56784 --dist-port 25684 --epmd-port 43691
.venv/bin/python -m tools.local_rabbit status --state-dir "$PWD/data/rabbit-local"
```

Передайте URL этого узла через окружение. Для локального узла, созданного helper, используются стандартный пользователь guest и virtual host `/`; пароль и полный URL не записывайте в общие артефакты. Путь вывода должен быть новым. Сам acceptance запускается только с явным `--run`:

```sh
.venv/bin/python -m tools.rabbit_acceptance --run \
  --output "$PWD/data/rabbit-acceptance-run-1" \
  --broker-control "$PWD/data/rabbit-local/control.json"
```

`control.json` содержит `scope=owned_loopback_broker`, loopback host, port и argv массивы `start`/`stop` (также поддерживаются `startargv`/`stopargv`). Адрес должен совпадать с URL. Скрипт выполняет только эти явно переданные команды без shell, с пределом 150 секунд. Helper отказывает для чужого каталога и не управляет `brew services`. Повторный start использует ту же ноду и Mnesia, что необходимо для проверки сохранности durable очереди. Во время прогона этот узел дважды останавливается; он должен быть выделен только этому прогону.

Проверяются следующие фазы:

1. Создание события edge → реальный publisher confirm → gateway SQLite → HTTP и WS. JPEG скачивается через API, проверяются SHA-256, формат и размер. Операторский PATCH виден в WS.
2. При остановленном consumer брокер отключается. Closure остаётся в edge outbox, ошибка sender видна. Брокер включается, sender восстанавливается; edge аварийно завершается сразу после реального publish-return до удаления outbox.
3. Consumer ещё остановлен. Подтверждённое persistent сообщение в durable очереди переживает restart брокера с прежней Mnesia.
4. Gateway аварийно завершается после SQLite commit до возврата callback и ACK. На диске одна строка события и две receipts; не ACKed доставка возвращается в очередь.
5. Gateway и edge перезапускаются. Повтор edge совпадает с сохранённым wire SHA, включая JPEG. Повторная доставка и старый event не создают новых строк, receipts или WS изменений. Closure, status и operator_note сохраняются; старый красный JPEG не заменяет новый синий.
6. Неподдерживаемый wire version получает настоящий permanent reject. Однократная storage ошибка в callback получает NACK/requeue, видна в состоянии consumer и восстанавливается без ручного рестарта. Повтор не меняет событие.

Аварийные барьеры находятся только в acceptance actors: wrapper вызывает настоящий publisher, затем `os._exit(72)` до `mark_published`; wrapper вызывает настоящий gateway receive/commit, затем `os._exit(73)` до callback-return/ACK. Production failpoints и fake broker не используются. Это подтверждает границы commit/ACK без вероятностного kill по таймеру.

Без `--broker-control` процессные проверки разрешены, но offline/restart брокера отмечаются `not_verified`, scope=`process-crash-only`, status=`passed_with_unverified_phases`. Это не полный acceptance. Отсутствующий RabbitMQ, timeout или ошибка дают ненулевой exit; отсутствие брокера не считается успешным тестом. `--timeout` ограничен 120 секундами на ожидание, default 20; actor RPC/остановка имеют отдельные пределы. Общий рабочий deadline — 600 секунд (`--deadline`, POSIX); cleanup имеет отдельные ограниченные ожидания. `/health=ok` не используется как единственное доказательство readiness: проверяются AMQP connected, реальные SQL/HTTP/WS результаты и publisher confirm.

`report.json` содержит версии Python/pika/RabbitMQ, порты, результаты фаз, wire/snapshot SHA, SQL counts и readback решения оператора. В каталоге остаются обе SQLite, JPEG, точный синтетический wire и безопасные actor logs. Не удаляйте их до аудита. Cleanup завершает только собственные subprocess, удаляет только очереди UUID этого запуска; общий exchange сохраняется. Переданный owned брокер в конце проверки останавливается; вручную его можно остановить той же командой:

```sh
.venv/bin/python -m tools.local_rabbit stop --state-dir "$PWD/data/rabbit-local"
```

Границы: не проверяются ML/CPU производительность, LinQ, настоящий RTSP, видео-клипы, preview/reconnect TTL и power-loss/fsync оборудования. Gateway не переносит edge clip; `clip_path` проверяется как отсутствующий. Preview ordering/TTL проверяются отдельно regression тестами runtime. Гарантия Event/outbox начинается с успешного SQLite commit; crash до первого успешного сохранения события не покрыт durable replay. Для transient/uncertain SQLite write есть отдельные `tests/test_edge_persistence.py`: prepared batch удерживается до commit, side effects происходят после него, прерывание retry видно и блокирует успешный stop до восстановления.

## Выполненная приёмка 24.10.2026

На коде `6ef2d73` полный scope `broker-and-process-crash` прошёл **7/7 фаз**: RabbitMQ 4.4.0, pika 1.4.4, Python 3.11.16. Сохранённый отчёт — `data/rabbit-acceptance-live-20261024-2/report.json`. Использованы пять последовательных экземпляров actor-процессов; запланированные exit 72/73 подтверждены, неожиданные nonzero exit дают ошибку. Итог: edge events=1/outbox=0, gateway events=1/receipts=2, ended_at=110, status=confirmed, операторская заметка и свежий JPEG сохранены. Повторы проверены точными message_id/completed-callback counters; телеметрия в эти счётчики не входит.

Первый запуск `data/rabbit-acceptance-live-20261024-1/report.json` сохранён как **failed**. Он обнаружил RabbitMQ 4.4 `INTERNAL_ERROR 541`: `transient_nonexcl_queues` запрещена. Исправление `429a174` использует `.telemetry.v2` с durable queue metadata, TTL/max-length и непостоянными сообщениями; deprecated feature не включалась. Также исправлен cleanup, скрывавший первичную channel-ошибку.

Cleanup успешного запуска удалил собственные UUID очереди, сохранил общий exchange и остановил брокер. Независимое read-only открытие обеих SQLite подтвердило counts/операторское решение; проверены SHA итогового снимка и отсутствие IPv4/IPv6 listeners обоих тестовых узлов. Общий отчёт интеграции — `.orchestra/metrics/local-validation-20261024.json`. Брокер установлен локально, системная служба и автозапуск не включались.

RabbitMQ 4.4 запрещает transient nonexclusive classic queues по умолчанию. Telemetry topology использует имя `guard.gateway.<edge_id>.telemetry.v2` и durable очередь с прежними TTL/max-length. Сами telemetry сообщения остаются неперсистентными (`delivery_mode=1`) и истекают; wire envelope version остаётся 1. Отдельное имя позволяет перейти с недолговечной v1 очереди без конфликта параметров. Обновлять edge и gateway нужно вместе. Старая `.telemetry` автоматически не удаляется; при миграции оператор может удалить её после остановки старых процессов. Acceptance удаляет только собственные текущие очереди UUID.
