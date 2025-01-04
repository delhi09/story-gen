# story-gen


## 仮想環境構築
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行方法
```sh
cp .env.sample .env
# .envにOpenAIのAPIキーを書く
python main.py
```

## 仮想環境解除
```sh
deactivate
```
## format
```sh
ruff check . --fix && ruff format .
```
