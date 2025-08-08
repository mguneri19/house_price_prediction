# 🏠 Ev Fiyat Tahmin Modeli

Bu proje, Kaggle'ın "House Prices - Advanced Regression Techniques" yarışması için geliştirilmiş bir makine öğrenmesi modelidir. Proje, ev fiyatlarını tahmin etmek için çeşitli özellikleri analiz eder ve LightGBM algoritması kullanarak yüksek performanslı tahminler üretir.

## 📋 Proje Özeti

- **Hedef**: Ev fiyatlarını tahmin etmek
- **Veri Seti**: Kaggle House Prices Competition
- **Model**: LightGBM Regressor
- **Performans**: RMSE ~22,118
- **Teknolojiler**: Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib, Seaborn

## 🚀 Özellikler

### Veri Analizi
- **EDA (Exploratory Data Analysis)**: Kapsamlı veri keşfi
- **Eksik Değer Analizi**: Eksik verilerin tespiti ve doldurulması
- **Outlier Analizi**: Aykırı değerlerin tespiti ve işlenmesi
- **Korelasyon Analizi**: Değişkenler arası ilişkilerin incelenmesi

### Feature Engineering
- **Yeni Özellikler**: Toplam alan, yaş hesaplamaları, kalite skorları
- **Kategorik Encoding**: Label encoding ve one-hot encoding
- **Özellik Seçimi**: Gereksiz değişkenlerin çıkarılması

### Modelleme
- **LightGBM Regressor**: Yüksek performanslı gradient boosting
- **Cross-Validation**: Model performansının değerlendirilmesi
- **Hiperparametre Optimizasyonu**: GridSearchCV ile optimizasyon
- **Feature Importance**: Özellik önem derecelerinin analizi

## 📊 Veri Seti

Proje, aşağıdaki özellikleri içeren ev verilerini kullanır:

- **Temel Özellikler**: Lot alanı, yaş, kat sayısı
- **Kalite Özellikleri**: Genel kalite, dış kalite, mutfak kalitesi
- **Alan Özellikleri**: Toplam alan, bodrum alanı, garaj alanı
- **Konum Özellikleri**: Mahalle, cadde tipi, arazi şekli

## 🛠️ Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/kullaniciadi/ev-fiyat-tahmin.git
cd ev-fiyat-tahmin
```

2. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Veri setini indirin:**
   - Kaggle'dan "House Prices - Advanced Regression Techniques" veri setini indirin
   - `datasets/housePrice/` klasörüne `train.csv` ve `test.csv` dosyalarını yerleştirin

4. **Projeyi çalıştırın:**
```bash
python house_prediction.py
```

## 📁 Proje Yapısı

```
ev-fiyat-tahmin/
├── house_prediction.py          # Ana model dosyası
├── datasets/
│   └── housePrice/
│       ├── train.csv           # Eğitim verisi
│       └── test.csv            # Test verisi
├── requirements.txt             # Gerekli kütüphaneler
├── README.md                   # Proje dokümantasyonu
└── .gitignore                  # Git ignore dosyası
```

## 📈 Model Performansı

- **RMSE**: 22,118.41
- **Model**: LightGBM Regressor
- **Özellik Sayısı**: 181
- **Train Veri**: 1,460 örnek
- **Test Veri**: 1,459 örnek

## 🎯 Kullanım

1. Veri setini `datasets/housePrice/` klasörüne yerleştirin
2. `python house_prediction.py` komutunu çalıştırın
3. Model eğitimi tamamlandıktan sonra `housePricePredictions.csv` dosyası oluşacaktır

## 📊 Sonuçlar

Proje tamamlandığında:
- Eğitilmiş model
- Feature importance grafikleri
- Kaggle submission dosyası (`housePricePredictions.csv`)
- Detaylı analiz raporları

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Geliştirici

**Muhammet Güneri**
- GitHub: [@mguneri19(https://github.com/mguneri19)
- LinkedIn: [www.linkedin.com/in/muhammet-guneri]

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! 
