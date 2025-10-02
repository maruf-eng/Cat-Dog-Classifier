# 🐱🐶 Cat vs Dog Classifier

## 📌 Proje Hakkında
Bu proje, **Global AI Hub – Akbank Derin Öğrenmeye Giriş Bootcamp** kapsamında geliştirilmiştir.  
Amaç, verilen bir görselin **kedi 🐱 mi yoksa köpek 🐶 mi olduğunu** tahmin eden bir derin öğrenme modeli tasarlamaktır.  

📒 Projeye Kaggle üzerinden buradan ulaşabilirsiniz:  
🔗 [Kedi ve Köpek Sınıflandırması Notebook](https://www.kaggle.com/code/berkekarakanl/kedi-ve-k-pek-s-n-fland-rmas)  

🌐 Kaggle Profilim: [Maruf KÜN](https://www.kaggle.com/marufkn)  

---

## ⚙️ Kullanılan Teknolojiler
- Python 🐍  
- TensorFlow / Keras  
- OpenCV  
- NumPy & Pandas  
- Matplotlib & Seaborn  

---

## 📂 Dataset
Proje, **[Kaggle Cats and Dogs Dataset](https://www.kaggle.com/datasets)** kullanılarak gerçekleştirilmiştir.  
Dataset Train / Validation / Test olarak ayrılmıştır.  

Veri artırma (Data Augmentation) teknikleri uygulanmıştır:  
- 📌 `rotation_range=20`  
- 📌 `zoom_range=0.2`  
- 📌 `horizontal_flip=True`  
- 📌 `rescale=1./255`  

Bu sayede modelin **overfitting** eğilimi azaltılmıştır.  

---

## 🏗️ Model Mimarisi

### 1️⃣ CNN (Scratch Model)
- Conv2D + MaxPooling2D blokları  
- Flatten + Dense katmanlar  
- Dropout → Overfitting’i azaltmak için  

### 2️⃣ Transfer Learning (EfficientNetB0)
- EfficientNetB0 (ImageNet ağırlıkları ile)  
- GlobalAveragePooling2D  
- Dense (128, ReLU) + Dropout(0.3)  
- Dense (1, Sigmoid) → Binary Classifier  

📌 **Model.summary(), Eğitim Süreci, Classification Report ve Sonuçlar:**

```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ efficientnetb0 (Functional)     │ (None, 8, 8, 1280)     │     4,049,571 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │       163,968 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           129 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,541,864 (17.33 MB)
 Trainable params: 164,097 (641.00 KB)
 Non-trainable params: 4,049,571 (15.45 MB)
 Optimizer params: 328,196 (1.25 MB) 

---

Eğitim Süreci:
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  
- Metric: Accuracy, Precision, Recall  
- Epoch: 20 (EarlyStopping ile durduruldu)  

Classification Report Çıktısı:

              precision    recall  f1-score   support

         Cat       0.00      0.00      0.00        48
         Dog       0.50      1.00      0.67        48

    accuracy                           0.50        96
   macro avg       0.25      0.50      0.33        96
weighted avg       0.25      0.50      0.33        96

---

🧾 Sonuçlar:
- Test setinde elde edilen doğruluk (Accuracy): %50  
- Precision, Recall ve F1-Score metrikleri raporlanmıştır.  
- Confusion Matrix görseli notebook içerisinde sunulmuştur.  
- Daha büyük dataset ve fine-tuning ile başarı %90+ seviyelerine çıkarılabilir.  

---

🔎 Model Açıklanabilirliği:
Modelin kararlarını açıklayabilmek için Grad-CAM (Gradient-weighted Class Activation Mapping) yöntemi uygulanmıştır.  
- Görseller üzerinde modelin hangi bölgeleri dikkate aldığı heatmap ile gösterilmiştir.  
- Böylece modelin karar mekanizması şeffaf hale getirilmiştir.  

---

🖼️ Test Örneği:
Bir test görseli için model çıktısı:  
- Tahmin Skoru: 0.87  
- 👉 Tahmin Sınıfı: Köpek 🐶  

---

🔮 Gelecek Çalışmalar:
- Daha büyük dataset ile eğitim  
- VGG16, ResNet gibi farklı transfer learning modelleri  
- Veri artırma (augmentation) yöntemlerini genişletme  
- Modeli web arayüzü veya mobil uygulamaya deploy etme  

---

💻 Nasıl Çalıştırılır:
git clone https://github.com/maruf-eng/Cat-Dog-Classifier.git  
pip install -r requirements.txt  
jupyter notebook cat-dog.ipynb
