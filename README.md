# FinaxioGPT: запуск приложения

Инструкция для локального запуска backend-сервиса RAG и веб-интерфейса. Backend отправляет запросы к NVIDIA API; для работы нужен API-ключ NVIDIA.

## 1. Установите Python

Установите Python версии 3.10 или новее с [python.org](https://www.python.org/downloads/). При установке в Windows включите опцию **Add Python to PATH**.

Проверьте установку:

```bash
python3 --version
```

В Windows вместо `python3` может использоваться команда:

```powershell
py --version
```

## 2. Перейдите в папку приложения

Откройте терминал в папке `server` — там находятся `rag.py`, `requirements.txt` и `index.html` — либо перейдите туда командой:

```bash
cd путь/к/finaxioGPT/server
```

## 3. Создайте и активируйте виртуальное окружение

Создание окружения:

```bash
python3 -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Если команда `python3` недоступна в Windows, используйте `py` при создании окружения:

```powershell
py -m venv .venv
```

## 4. Установите зависимости

При активном виртуальном окружении выполните:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

В `requirements.txt` указан CPU-вариант PyTorch. Первая установка может занять некоторое время.

## 5. Настройте ключ NVIDIA API

Backend ищет ключ в следующем порядке: переменная окружения `NVIDIA_API_KEY`, переменная `LLM_API_KEY`, затем поле `apiKey` в файле `api.md` в папке `server`. Поэтому для локального запуска можно использовать уже имеющийся `api.md` и не задавать переменную окружения.

Если файла `api.md` нет, создайте его в папке `server` со строкой:

```text
apiKey: ваш_api_ключ
```

При необходимости в этом же файле можно указать модель эмбеддингов:

```text
model: nvidia/nemotron-3.5-embed-8b
```

Не добавляйте файл с настоящим ключом в Git и не публикуйте его. Альтернативно можно задать ключ через переменную окружения; она имеет приоритет над `api.md`.

Linux/macOS (в текущем окне терминала):

```bash
export NVIDIA_API_KEY="ваш_api_ключ"
```

Windows PowerShell (в текущем окне):

```powershell
$env:NVIDIA_API_KEY = "ваш_api_ключ"
```

Переменная должна быть задана в том же терминале, из которого запускается backend.

## 6. Запустите backend

Из папки `server`, с активным виртуальным окружением и настроенным ключом в `api.md` или переменной окружения:

```bash
python rag.py
```

По умолчанию API будет доступен на `http://localhost:3050`. При запуске сервис инициализирует RAG; первый запуск может занять время. Оставьте это окно терминала открытым.

## 7. Запустите веб-интерфейс

Откройте **второе** окно терминала, перейдите в ту же папку `server` и запустите простой HTTP-сервер:

```bash
python3 -m http.server 8000
```

На Windows, если `python3` не распознаётся, выполните:

```powershell
py -m http.server 8000
```

Откройте в браузере:

```text
http://localhost:8000
```

Страница `index.html` отправляет запросы на `http://localhost:3050/send-request`, поэтому backend и сервер интерфейса должны работать одновременно. Для остановки каждого сервера нажмите `Ctrl+C` в соответствующем окне терминала.

## Запуск в следующий раз

Откройте два терминала в папке `server`. В первом активируйте окружение и запустите backend (если ключ хранится в `api.md`, задавать переменную не требуется):

```bash
source .venv/bin/activate
# Необязательно, если ключ уже записан в api.md:
export NVIDIA_API_KEY="ваш_api_ключ"
python rag.py
```

Во втором запустите UI-сервер:

```bash
python3 -m http.server 8000
```

На Windows для активации и настройки ключа используйте PowerShell-команды из разделов выше.
