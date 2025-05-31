# encoding=utf-8
# Programmer: 411203479, 李冠諭
# Date: 2025/05/25
# Python 影像分類
# 框架：Tensorflow, Keras
# Miniconda 3, Python 3.8
# Github: https://github.com/gregfluffy/Final-Fruits-Ripeness-Classification
"""
在終端機執行(Execute in terminal)
1. conda activate <your_ProjectName>
2. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
3. pip install "tensorflow==2.10"
4. pip install -r requirements.txt
"""
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# 定義資料路徑及類別
data_dir = 'dataset/train'
categories = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples',
              'rottenbanana', 'rottenoranges', 'unripe apple', 'unripe banana', 'unripe orange']

def load_data(data_dir, categories):
    """讀取影像並返回資料和標籤"""
    data, labels = [], []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_array = cv2.resize(img_array, (150, 150))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return np.array(data), np.array(labels)

# 讀取資料
data, labels = load_data(data_dir, categories)
print(f"Total images loaded: {len(data)}")

# 預處理資料
data = preprocess_input(data)
labels = to_categorical(labels, num_classes=len(categories))

# 繪製圓餅圖
class_counts = np.bincount(np.argmax(labels, axis=1))
plt.figure(figsize=(8, 8))
plt.pie(class_counts, explode=[0.1]*len(categories), labels=categories, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Image Classes')
plt.axis('equal')
plt.show()

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 載入 EfficientNet 預訓練模型
base_model = EfficientNetB0(weights='imagenet', include_top=False,
                            input_shape=(150, 150, 3))
base_model.trainable = False

# 建立完整模型
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# 定義 EarlyStopping 回調
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=5,
                               restore_best_weights=True)

# 訓練模型
history = model.fit(X_train, y_train,
                    epochs=30,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# 繪製訓練和驗證的準確率和損失
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# 將訓練和驗證的準確率和損失轉換為 DataFrame
history_df = pd.DataFrame(history.history)

# 預測測試集
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 混淆矩陣和分類報告
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes, target_names=categories)

# 顯示結果
print('Confusion Matrix')
print(conf_matrix)
print('\nClassification Report')
print(class_report)

# 可視化類別
class Visualization:
    def __init__(self, vis_datas, n_ims, rows, cls_names=None, cls_counts=None):
        self.vis_datas = vis_datas
        self.n_ims = n_ims
        self.rows = rows
        self.cls_names = cls_names
        self.cls_counts = cls_counts
        self.visualize()
    def visualize(self):
        cols = self.n_ims // self.rows
        plt.figure(figsize=(15, 10))
        for i in range(self.n_ims):
            plt.subplot(self.rows, cols, i + 1)
            img, label = self.vis_datas[i]  # 假設 vis_datas 是一個包含 (圖像, 標籤) 的元組
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換為 RGB
            plt.imshow(img_rgb)  # 不需要指定 cmap，因為這是 RGB 圖像
            plt.title(self.cls_names[label])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# tr_dl 和 ts_dl 是訓練和測試數據，並且已經加載了圖像和標籤
# 這裡使用 X_train 和 y_train 作為示例數據
vis_datas = [(X_train[i], np.argmax(y_train[i])) for i in range(min(18, len(X_train)))]  # 取前 18 張圖像
vis = Visualization(vis_datas=vis_datas, n_ims=18, rows=6, cls_names=categories, cls_counts=class_counts)
