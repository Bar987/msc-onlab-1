# Mélymegerősítéses tanulás kirakós játékokhoz

## Függőségek telepítése

A kód Python 3.7-et használ.

```
pip install -r requirements.txt
```

## Tanítás indítása

```
python3 train.py
```

## Környezet kipróbálása

```
python3 poc.py
```

### Egy 3x3-as példa az action space-re

A tile-ok közötti szám, mint input, a két szomszédos tile cseréjét eredményezi.

![alt text](./figs/env2.png '3x3')

### Colab notebook

https://colab.research.google.com/drive/1BRKmObPp4Bk0FoBSIH4bxwkpHSEB_qM6?usp=sharing

A tanítás kb. 45 percet vesz igénybe a GPU-val rendelkező környezetben.
