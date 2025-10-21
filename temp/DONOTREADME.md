projekt/
│
├── data/
│   ├── raw/                # dane surowe (np. CSV z zewnątrz)
│   │   └── iris_raw.csv
│   └── processed/          # dane po wstępnym czyszczeniu
│       └── iris_clean.csv
│
├── notebooks/              # notebooki Jupyter do eksploracji
│   └── eda.ipynb
│
├── src/                    # kod źródłowy Pythona
│   ├── __init__.py
│   ├── data_loader.py      # funkcje do wczytywania danych
│   ├── preprocess.py       # czyszczenie, standaryzacja danych
│   ├── model.py            # trenowanie i testowanie modelu
│   └── utils.py            # funkcje pomocnicze
│
├── outputs/                # wyniki eksperymentów
│   ├── models/
│   │   └── model.pkl       # zapisany wytrenowany model
│   └── figures/
│       └── confusion_matrix.png
│
├── main.py                 # główny plik uruchamiający pipeline ML
├── requirements.txt        # lista bibliotek
├── README.md               # opis projektu
└── .gitignore              # ignorowane pliki