# Презентация «Зазеркалье»: чек-поинт 3 и защита

Итоговый файл — **`zazerkalye.pdf`**: 16:9, 1920×1080, один слайд на страницу. 12 основных слайдов по структуре `docs/PITCH.md` и 4 резервных (Б1–Б4) для вопросов.

| Файл | Что внутри |
|---|---|
| `index.html` | разметка слайдов и тексты |
| `deck.css` | стили: тёмно-синий + салатовый, «окна» и пиксельные детали по мотивам кейса |
| `deck.js` | хром и подвал слайдов, подстановка данных, SVG-графики (без внешних библиотек) |
| `data.json` | **цифры для графиков и плейсхолдеров** — правится без вёрстки |
| `render.mjs` | сборка PDF и PNG через Playwright, проверка переполнений |
| `fonts/` | Tektur, Onest, JetBrains Mono (SIL OFL 1.1, лицензии рядом). CDN не используется |

Скриншоты берутся из `../screens/` (не копируются).

## Как пересобрать PDF

Нужны Node 20 и Playwright с Chromium. Playwright есть в зависимостях фронтенда.

```bash
# один раз: зависимости фронтенда и браузер для Playwright
cd frontend && npm ci && npx playwright install chromium && cd ..

# сборка PDF — через сторож памяти, браузер запускается один и закрывается сам
tools/heavy.sh node deliverables/presentation/render.mjs

# PDF + PNG каждого слайда для проверки вёрстки
tools/heavy.sh node deliverables/presentation/render.mjs --png /tmp/zz-preview
```

Если `frontend/node_modules` нет, укажите Playwright из другого места:
`PLAYWRIGHT_DIR=/путь/к/node_modules/playwright node deliverables/presentation/render.mjs`.

Скрипт поднимает локальный HTTP-сервер на `deliverables/`, ждёт загрузки шрифтов и `data.json`, проверяет картинки и вёрстку и печатает предупреждения: «ниже области контента на N px», «обрезан текст». После правок добейтесь строки `Вёрстка: переполнений не найдено` и посмотрите PNG.

Посмотреть слайды в браузере: `node deliverables/presentation/render.mjs --serve` и открыть ссылку из консоли. При открытии `index.html` двойным кликом (`file://`) браузер не даст загрузить `data.json` — страница покажет красную плашку.

## Как обновить цифры к защите

Всё, что появится позже, вынесено в `data.json → pending`. Пока `status: "pending"`, на слайде стоит розовая метка **«будет обновлено к защите»** и пунктирная рамка. Чтобы вставить замер, впишите значения, поставьте `"status": "done"` и `measured_at`, затем пересоберите PDF. Метка сменится на «замер <дата>», рамка — на обычную.

| Ключ в `data.json` | Слайд | Что вписать |
|---|---|---|
| `pending.eval.metrics[].value` | 6 «Надёжность тревог» | precision и recall по событиям, дубли, ложные тревоги в час, задержка p50 — из `metrics/eval-*.json`. Можно поменять `dataset`, если роликов станет больше |
| `pending.edge_latency.p50_s`, `p95_s` | 5 «Демо», 7 «LinQ TPU» | сквозная задержка «начало нарушения → карточка в UI» в режиме edge + gateway на стенде |
| `pending.edge_mode` | 4 «Как это работает» | когда режим edge заработает вживую — `status: "done"`; текст берётся из `done_text` |
| `scenarios.rows[].state`, `status`, `scenarios.as_of` | Б3 «Сценарии» | `live` — работает в живом конвейере, `wip` — в работе, `ready` — плагин и тесты готовы |

Пример для задержки:

```json
"edge_latency": { "status": "done", "p50_s": 2.6, "p95_s": 3.1, "measured_at": "25.10 03:00", ... }
```

Графики TPU против CPU (`tpu_vs_cpu`), окна det 480 (`det480_windows`) и «камер на TPU» (`cameras_per_tpu`) тоже строятся из `data.json`. Если замеры переснимут, поправьте значения там и источник в `docs/BENCHMARKS.md`.

## Откуда цифры

Каждая цифра — из документов репозитория, источник подписан внизу слайда.

| Слайд | Источник |
|---|---|
| 1 Титул | `deliverables/screens/real-02-event-card.jpg` — живой бэкенд, local, CPU |
| 2 Проблема | кейс LinQ, слайды 2–3, 7; Росстат 2025 — `docs/ANALYTICS.md` §1.3 |
| 3 Для кого | `docs/ANALYTICS.md` §1.1–1.2 |
| 4 Как это работает | `docs/ARCHITECTURE.md` §1–3, `docs/BENCHMARKS.md` §4.3 |
| 5 Демо | `deliverables/screens/real-*.jpg`, `docs/BENCHMARKS.md` §5 |
| 6 Надёжность | `docs/ARCHITECTURE.md` §5; eval — плейсхолдер |
| 7 LinQ TPU | `docs/BENCHMARKS.md` §1–3 |
| 8 Edge и камеры | `docs/BENCHMARKS.md` §4–5 |
| 9 Без кода | `README.md` «Как добавить новый тип нарушения»; скриншот `05b-zone-drawing.jpg` — **мок-режим**, на слайде так и подписан |
| 10 Рынок | `docs/ANALYTICS.md` §3 |
| 11 Экономика | `docs/PITCH.md` «Экономика внедрения»; суммы — условные допущения Д-n |
| 12 Что дальше | `docs/PITCH.md`, `docs/ARCHITECTURE.md` §6–7 |
| Б1–Б2 | `docs/BENCHMARKS.md` §1, §4 |
| Б3 | `README.md` «Сценарии и статус» |
| Б4 | `docs/CHECKPOINT_2.md` §4, `README.md` «Модели и лицензии» |

Скриншоты без префикса `real-` сняты в мок-режиме UI. Их цифры замерами не подаём.
