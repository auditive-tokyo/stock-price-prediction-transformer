# 4本値CSVをAPIから取得、CSVからチャート作成、ニュースや板情報をスクレイピングして、AIによる自動売買を行うPythonアプリです。

---

## 必要なもの

- Python 3.12.10
- Mac または Windows（PowerShell）

---

## 1. Pythonのインストールと確認

### Macの場合

1. ターミナルを開きます。
2. Pythonがインストールされているか確認します：

    ```sh
    python --version
    ```

    → `Python 3.12.10` と表示されればOKです。

    表示されない場合は任意の方法で、インストールしてください。Homebrewからpyenvインストールするなど

### Windows（PowerShell）の場合

1. PowerShellを開きます。
2. Pythonがインストールされているか確認します：

    ```powershell
    python --version
    ```

    → `Python 3.12.10` と表示されればOKです。

    表示されない場合は[公式サイト](https://www.python.org/downloads/release/python-31210/)からインストールしてください。

---

## 2. 仮想環境の作成と有効化

### Mac

```sh
python -m venv venv
source venv/bin/activate
```

### Windows（PowerShell）
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 3. 必要なライブラリのインストール

（Mac・Windows共通）

```sh
pip install -r requirements.txt
```

## 4. 板情報の自動取得（15分ごと）
```
python SBI証券から15分毎に板情報を取得する.py
```

## 5. メインフローの実行（AIによる自動売買判断、15分ごと）

別のターミナル（またはPowerShell）で、同じ仮想環境を有効化した上で下記を実行してください。

```
python main.py
```