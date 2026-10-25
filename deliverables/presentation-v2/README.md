# Презентация к защите, версия 2: короткая и редактируемая

Итог — **`zazerkalye-v2.pptx`** (16:9, 10 слайдов) и **`zazerkalye-v2.pdf`**. Все элементы PPTX нативные: текстовые блоки, фигуры схемы и картинки. Их можно править в PowerPoint, Keynote или LibreOffice. У каждого слайда есть заметки докладчика на 2–4 фразы. Основа заметок — `docs/DEMO_SCRIPT.md`.

Прежняя колода на 25 слайдов (`deliverables/presentation/`) не менялась.

| # | Слайд | Картинка |
|---|---|---|
| 1 | Зазеркалье — видеоаналитика охраны труда на LinQ TPU | кадр события с рамкой и зоной |
| 2 | Нарушение → оповещение за 2 секунды | схема: камеры → edge + LinQ TPU → событие → оператор |
| 3 | Вся смена на одном экране | «Мониторинг» |
| 4 | Одно нарушение — одна карточка (+ RabbitMQ) | карточка события, поток «Событие → RabbitMQ → MES · SCADA · ERP · бот», строка подключения (exchange guard.edge.v1) |
| 5 | Журнал — доказательная база | «Журнал» |
| 6 | Зоны и правила — мышью, без кода | редактор зон |
| 7 | Где, когда и кто нарушает | «Аналитика» |
| 8 | Маховик данных: модели учатся на своём цехе | «Данные» |
| 9 | Всё работает на LinQ TPU | «Производительность» в режиме TPU и три цифры |
| 10 | Спасибо! | — |

## Файлы

| Файл | Что внутри |
|---|---|
| `slides.json` | тексты слайдов, заметки докладчика, какие скриншоты и как обрезать |
| `build.py` | вёрстка (python-pptx), PDF через LibreOffice, PNG для проверки, проверка переполнения текста по метрикам Onest |
| `capture.mjs` | скриншоты интерфейса (Playwright, окно 1600×900, масштаб 1.5; вход admin/admin), только чтение |
| `screens/` | скриншоты, на которые ссылается `slides.json` |
| `assets/` | знак и жёлто-чёрная штриховка; `build.py` создаёт их, если файлов нет |
| `fonts.conf` | каталоги шрифтов для LibreOffice: без него в PDF вместо Onest попадает Linux Libertine |

Цвета взяты из `frontend/src/styles/global.css`, шрифт — Onest (`~/Library/Fonts`). Если на компьютере показа Onest нет, PowerPoint подставит системный шрифт без засечек. PDF не зависит от шрифтов компьютера.

## Пересборка

```bash
# 1. Скриншоты: приложение на http://127.0.0.1:8000, вход admin/admin, если включён
tools/heavy.sh /opt/homebrew/bin/node deliverables/presentation-v2/capture.mjs
#    только часть экранов, отдельный каталог:
SHOTS_DIR=/tmp/shots tools/heavy.sh /opt/homebrew/bin/node deliverables/presentation-v2/capture.mjs --only event,zones

# 2. PPTX, PDF и PNG для проверки
tools/heavy.sh .venv/bin/python deliverables/presentation-v2/build.py --pdf --png /tmp/v2-png
```

`capture.mjs` ничего не сохраняет и не меняет статусы. Баннер критической тревоги на скриншотах скрыт стилем: в демо ролики зациклены, и баннер всплывает каждые несколько секунд. Чтобы его оставить, задайте `KEEP_ALARM=1`. Экраны: `monitoring`, `event`, `journal`, `zones`, `rules`, `analytics`, `employees`, `data`, `performance`.

`build.py` печатает «Вёрстка: переполнений не найдено» или список блоков, где текст не помещается. Обрезка картинок (`crop` в `slides.json`) нативная, в PowerPoint её можно снять.

## Источники цифр

- «≈2 с», «4 × 4,9 к/с», «×4 против CPU» — `docs/EDGE.md` §6.1 и §6.3, замер `edge-tpu-20261024-1002.json`: сквозная задержка p50 2,20 с, 4,8–4,9 к/с на камеру, 19,4 против 4,8 к/с на CPU того же контейнера.
- «13 типов нарушений» — плагины `guard/rules/plugins/`, `GET /api/rules/types`.
- Типы зон — `frontend/src/domain/labels.ts`, контракт выбора зон — `docs/RULE_ZONE_METADATA.md`.
- RabbitMQ — `docs/EDGE_PROTOCOL.md`: JSON-конверт, durable outbox, идемпотентное применение.
- Сотрудники — `docs/EMPLOYEES.md`, маховик — `docs/FLYWHEEL.md`.
