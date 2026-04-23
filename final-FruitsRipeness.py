# encoding=utf-8
# Programmer: 411203479, 李冠諭
# Date: 2025/05/25
# Python 影像分類
# 框架：TensorFlow, Keras
# Miniconda 3, Python 3.8
# GitHub: https://github.com/gregfluffy/Final-Fruits-Ripeness-Classification
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
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# 定義資料路徑及類別
DATA_DIR = 'dataset'
CATEGORIES = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples',
              'rottenbanana', 'rottenoranges', 'unripe apple', 'unripe banana', 'unripe orange']


def load_data(path_to_data, category_list):
    """讀取影像並返回資料和標籤"""
    data_list, label_list = [], []
    for category in category_list:
        path = os.path.join(path_to_data, category)
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue

        class_num = category_list.index(category)
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)

                # 2. 檢查圖片是否讀取成功，解決 OpenCV 類型警告
                if img_array is not None:
                    resized_array = cv2.resize(img_array, (150, 150))
                    data_list.append(resized_array)
                    label_list.append(class_num)
                else:
                    print(f"Skipping broken image: {img_name}")
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(data_list), np.array(label_list)


# 執行讀取
data, labels = load_data(DATA_DIR, CATEGORIES)
print(f"Total images loaded: {len(data)}")

# 預處理資料
data = preprocess_input(data)
labels = to_categorical(labels, num_classes=len(CATEGORIES))

# 繪製圓餅圖
class_counts = np.bincount(np.argmax(labels, axis=1))
plt.figure(figsize=(8, 8))
plt.pie(class_counts, explode=[0.1] * len(CATEGORIES), labels=CATEGORIES, autopct='%1.1f%%', startangle=140)
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
    Dense(len(CATEGORIES), activation='softmax')
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
class_report = classification_report(y_true, y_pred_classes, target_names=CATEGORIES)

# 顯示結果
print('Confusion Matrix')
print(conf_matrix)
print('\nClassification Report')
print(class_report)

# 混淆矩陣圖像化
def plot_confusion_matrix(matrix, labels):
    plt.figure(figsize=(10, 8))
    # 使用 seaborn 繪製熱圖
    # annot=True 顯示數字, fmt='d' 格式為整數, cmap 選擇顏色 (如 'Blues', 'Greens', 'YlGnBu')
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.title('Fruit Ripeness Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)  # 旋轉標籤避免重疊
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 顯示圖像
plot_confusion_matrix(conf_matrix, CATEGORIES)

# 可視化類別
class Visualization:
    # 建議將傳入參數名稱稍微修改，避免與外部變數混淆
    def __init__(self, data_samples, n_ims, rows, cls_names=None):
        self.vis_datas = data_samples
        self.n_ims = n_ims
        self.rows = rows
        self.cls_names = cls_names
        self.visualize()

    def visualize(self):
        cols = self.n_ims // self.rows
        plt.figure(figsize=(15, 10))
        for i in range(min(self.n_ims, len(self.vis_datas))):
            plt.subplot(self.rows, cols, i + 1)
            img, label = self.vis_datas[i]

            # 注意：EfficientNet 預處理過的圖片值會變動，
            # 如果 img 是經過 preprocess_input 的，直接顯示顏色會變得很奇怪
            img_min, img_max = img.min(), img.max()
            img_display = (img - img_min) / (img_max - img_min) * 255
            img_rgb = cv2.cvtColor(img_display.astype('uint8'), cv2.COLOR_BGR2RGB)

            plt.imshow(img_rgb)
            plt.title(self.cls_names[label] if self.cls_names else label)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


# 呼叫視覺化
vis_samples = [(X_train[i], np.argmax(y_train[i])) for i in range(min(18, len(X_train)))]
vis = Visualization(data_samples=vis_samples, n_ims=18, rows=3, cls_names=CATEGORIES)
