import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping  # 加入 EarlyStopping

# 定義資料路徑
data_dir = 'dataset/test'
categories = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges', 'unripe apple', 'unripe banana', 'unripe orange']

# 初始化資料和標籤列表
data = []
labels = []

# 讀取影像並調整大小
for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in tqdm(os.listdir(path)):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (150, 150))  # EfficientNet 預設大小為 150x150
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            pass

print(f"Total images loaded: {len(data)}")

# 將資料轉換為 NumPy 陣列並預處理
data = np.array(data)
data = preprocess_input(data)  # 使用 EfficientNet 的預處理函數
labels = np.array(labels)

# 將標籤轉換為 one-hot 編碼
labels = to_categorical(labels, num_classes=len(categories))

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 載入 EfficientNet 預訓練模型
base_model = EfficientNetB0(weights='imagenet',
                            include_top=False,
                            input_shape=(150, 150, 3))

# 冻結預訓練模型的權重
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
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 定義 EarlyStopping 回調
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=3,
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
