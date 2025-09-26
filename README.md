# 🐱🐶 Cat vs Dog Classifier

## 📌 Proje Hakkında
Bu proje, **derin öğrenme tabanlı görüntü sınıflandırma** problemi üzerine geliştirilmiştir.  
Amaç, verilen bir görselin **kedi 🐱 mi yoksa köpek 🐶 mi olduğunu** tahmin eden bir yapay zeka modeli tasarlamaktır.  

---

## ⚙️ Kullanılan Teknolojiler
- Python 🐍  
- TensorFlow / Keras  
- OpenCV  
- NumPy & Pandas  
- Matplotlib & Seaborn  

---

## 📂 Dataset
Proje, Kaggle’dan alınan **Cat & Dog Dataset** ile gerçekleştirilmiştir.  
Dataset Train / Validation / Test olarak ayrılmıştır.  

🔗 [Kaggle Dataset](https://www.kaggle.com/datasets)  

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

---

## 📊 Eğitim Süreci
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metric:** Accuracy, Precision, Recall  
- **Epoch:** 20 (EarlyStopping ile durduruldu)  

📈 Eğitim ve doğrulama sonuçları grafiklerle görselleştirilmiştir:  
_(örneğin: loss ve accuracy grafikleri buraya eklenebilir)_  

---

## 🧾 Sonuçlar
- Accuracy ≈ %55–60 (küçük dataset nedeniyle sınırlı başarı)  
- Model bazı örneklerde doğru, bazılarında yanlış sınıflandırma yapmıştır.  
- Daha büyük dataset ve fine-tuning ile başarı **%90+** seviyelerine çıkabilir.  

---

## 🖼️ Test Örneği
Aşağıdaki örnek modelin tahmin çıktısını göstermektedir:  

- **Tahmin Skoru:** `0.87`  
- 👉 **Tahmin Sınıfı:** Köpek 🐶  

---

## 🔮 Gelecek Çalışmalar
- Daha büyük dataset ile eğitim  
- VGG16, ResNet gibi farklı transfer learning modelleri  
- Veri artırma (augmentation) yöntemlerini genişletme  
- Modeli web arayüzü veya mobil uygulamaya deploy etme  

---
## 💻 Nasıl Çalıştırılır

```bash
git clone https://github.com/berkekarakanli/Cat-Dog-Classifier.git
pip install -r requirements.txt
jupyter notebook cat-dog.ipynb



