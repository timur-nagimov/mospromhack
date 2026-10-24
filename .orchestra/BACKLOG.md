# Следующие шаги

Обновлено 24.10.2026. Проверенный срез `6d2696d` для `main` включает 13 правил,
constructor validation, статусы наблюдаемости, zone metadata и редактор камеры;
приёмка — 2099 backend + 12 frontend тестов и production build.
[Отчёт](metrics/violation-expansion-20261024.json). Приёмка нового кода
из `origin/main` выполняется отдельно после интеграции.

Оставляю следующие задачи в порядке приоритета. Новые исследования и тяжёлые
прогоны сейчас не запускаю; каждая задача требует отдельного измеримого результата.

| Приоритет | Задача | Условие приёмки |
|---|---|---|
| 1 | Независимая видеопроверка wrong_way, zone_dwell и zone_overcapacity | Разметка camera/site split, uncertain/visibility и clean exposure; per-rule TP/FP/FN/duplicates, latency и coverage без подбора порогов по test. |
| 2 | Specialist PPE для очков и перчаток | Явные positive/negative классы; проверенные назначение СИЗ человеку и видимость; present/absent/unknown разметка, независимый quality gate. Имя класса само по себе не означает поддержку нарушения. |
| 3 | Capability validation для новых PPE правил | Общий CPU/LinQ контракт classes → реализованный attribute adapter → требования params; unknown/unsupported блокирует включение нового правила, старые конфиги сохраняют семантику. |
| 4 | Приёмка последней версии на внешнем LinQ | Закреплённые model/program/tail hashes, compile/INT8 parity и качество, реальные события edge → gateway → HTTP/WS, ошибка/восстановление и измеренные denominators. |
| 5 | Длительная эксплуатация | Проверить reconnect/capture boundaries, SQLite/outbox recovery, readiness, retention, лимиты RAM/диска; отдельно определить доставку edge-клипов на gateway. |

Для S1 нужна actor-aware связь событий и независимые clean-видео с размеченными
зонами; результат 3 TP / 1 FN / 1 duplicate на двух demo-видео не заменяет эту
проверку. Для новых PPE не добавляю «отсутствие обуви» без negative класса,
не подменяю респиратор классом `no_mask` и не считаю height strict распознаванием привязи.

Сохраняю решения предыдущих сравнений: full-frame PPE вместо ROI
([A/B](metrics/ppe-roi-quality-20261024.json)); текущий runtime вместо RF-DETR2XL,
у которого raw parity failed и преимущество по общему качеству не установлено
([quality](metrics/rfdetr-quality-20261024.json)). Повторные исследования должны
сохранять исходные GT/пороги и отделять event quality от CPU/TPU производительности.

Точки будущего capability-пакета и границы текущей интеграции описаны в
[VIOLATION_API_INTEGRATION.md](../docs/VIOLATION_API_INTEGRATION.md).
