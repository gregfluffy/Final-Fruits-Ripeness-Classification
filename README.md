# 🥭 Fruits-Ripeness-Classification

水果成熟度影像分類：判斷水果為「未熟、成熟或腐爛」。本專案使用 Kaggle 公開資料集與深度學習模型進行訓練與推論。

## 📌 專案簡介
本專案透過影像分類技術，自動判別水果的成熟度狀態，可應用於農業自動檢測、食品篩選、教育示範等場景。

## 🧠動機
傳統上，成熟度的判斷多仰賴人眼目測，然而這樣的方法主觀性高、效率低，也容易受到光線、經驗等因素影響。隨著人工智慧與電腦視覺技術的發展，若能透過影像自動分類水果為「未熟」、「成熟」與「腐爛」三類，不僅可提升判斷的準確度，也能大幅減少人力成本與食材浪費。

## 📊 數據來源

- 📂 Kaggle 資料集：  
  [Fruit Ripeness - Unripe, Ripe and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)

- 📄 參考程式碼（PyTorch）：  
  [F1 0.98 Fruits Ripeness Classification (Torch)](https://www.kaggle.com/code/killa92/f1-0-98-fruits-ripeness-classification-torch)

## 🖥️ 開發環境

| 工具項目 | 說明 |
|----------|------|
| IDE      | PyCharm |
| Python   | 3.8 |
| 環境管理 | Miniconda 3 |
| 編碼格式 | UTF-8 |
| 框架     | Tensorflow, Keras |

## 💡 演算法
### 演算法： EfficientNetB0  
### 核心概念:
EfficientNet是一種高效的卷積神經網絡（CNN）架構，其核心思想是通過複合縮放方法，同時調整網絡的深度、寬度和解析度，以達到最佳性能和資源利用率。 
EfficientNet使用了一種稱為MBConv的輕量級卷積塊，並通過自動化方法平衡網絡的三個維度，從而在保持模型高效運行的同時，顯著提升模型的識別精度。  
- 🧰	Keras API： [EfficientNetB0 function](https://keras.io/api/applications/efficientnet/#efficientnetb0-function)

## 📁 資料夾結構
資料集採用「類別分資料夾」的方式進行管理：
```
dataset/
 ├── freshapples/
 ├── freshbanana/
 ├── freshorange/
 ├── rottenapples/
 ├── rottenbanana/
 ├── rottenorange/
 ├── freshmango/
 ├── rottenmango/
 └── freshtomato/
```

## 安裝套件(需下載requirements.txt)
### 在終端機執行(Execute in terminal)
1. conda activate <your_ProjectName>  
2. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1  
3. pip install "tensorflow==2.10"  
4. pip install -r requirements.txt

### requirements.txt內包含以下套件  
- matplotlib==3.7.5
- numpy==1.24.4
- opencv-python==4.11.0.86
- pandas==2.0.3
- scikit-learn==1.3.2
- seaborn==0.13.2
- tqdm==4.67.1

## 專題簡報
- 📝 Canva簡報：  
  [人工智慧自主學習專題-水果成熟度影像分類](https://canva.link/qd5trlg4hv6vrql)
