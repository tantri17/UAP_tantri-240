# ML Streamlit Dashboard

UAP Praktikum Semester 7  
Nama: Tantri Romadhoni Siswining Ndaru  
NIM: 202210370311240  


## 🎵 Sistem Klasifikasi Citra Alat Musik Berbasis Deep Learning

Proyek ini merupakan implementasi Sistem Klasifikasi Citra Alat Musik yang dibangun dalam bentuk dashboard interaktif menggunakan Streamlit.
Sistem ini digunakan untuk mengklasifikasikan gambar alat musik ke dalam beberapa kelas, yaitu gitar, piano, drum, biola, cello dan saxophone, berdasarkan citra yang diunggah oleh pengguna.

Pada proyek ini digunakan tiga pendekatan model Deep Learning yang berbeda, yaitu:

CNN Base (Non-Pretrained)

MobileNetV2 (Pretrained Model)

EfficientNetB0 (Pretrained Model)

Ketiga model tersebut digunakan untuk melihat perbedaan karakteristik model, performa prediksi, serta confidence hasil klasifikasi citra.

-----------------------------------------------------------------------------
## 📂 Struktur Repository

```bash
ML_STREAMLIT_DASHBOARD/
│
├── dataset/
│   ├── train/                      # Data latih citra alat musik
│   ├── test/                       # Data uji citra alat musik
│   └── valid/                      # Data validasi citra alat musik
│
├── training/
│   ├── train_cnn.py                # Script training CNN dari awal (Scratch)
│   ├── train_mobilenet.py          # Script training MobileNetV2 (Transfer Learning)
│   └── train_efficientnet.py       # Script training EfficientNetB0 (Transfer Learning)
│
├── evaluation/
│   ├── eval_cnn.py                 # Evaluasi model CNN
│   ├── eval_mobilenet.py           # Evaluasi model MobileNetV2
│   └── eval_efficientnet.py        # Evaluasi model EfficientNetB0
│
├── models/
│   ├── cnn_scratch.h5              # Model CNN hasil training
│   ├── mobilenetv2.h5              # Model MobileNetV2 hasil fine-tuning
│   └── efficientnetb0.h5           # Model EfficientNetB0 hasil fine-tuning
│
├── app.py                          # Aplikasi utama Streamlit
└── README.md                       # Dokumentasi proyek

```
-----------------------------------------------------------------------------
## 📊 Dataset yang Digunakan (Sumber dataset universe.roboflow.com)


Dataset yang digunakan merupakan dataset citra alat musik yang disusun dalam struktur folder berdasarkan kelas (Image Classification format).
dapat di unduh lewat link berikut : 


https://universe.roboflow.com/music-instrument-recognition/musical-instrument-recognition/dataset/4/download 

📍 Kelas Alat Musik

Gitar

Piano

Drum

Biola

Saxophone

cello

Dataset dibagi menjadi:

Train → data latih

Validation → data validasi

Test → data pengujian


----------------------------------------------------------------------------
## Image Classification with CNN, EfficientNetB0, and MobileNetV2

Project ini berisi implementasi klasifikasi citra menggunakan tiga pendekatan model deep learning, yaitu:

CNN dari nol (scratch)

EfficientNetB0 (transfer learning)

MobileNetV2 (transfer learning)

Dataset dibagi menjadi train, validation, dan test, serta dilakukan training dan evaluasi terpisah untuk setiap model.


-----------------------------------------------------------------------------
## Preprocessing Data

Tahapan preprocessing citra yang dilakukan adalah sebagai berikut:

1. **Resize Gambar**  
   Semua gambar diubah ukurannya menjadi **224 × 224 piksel** agar sesuai dengan input model.

2. **Normalisasi Pixel**  
   Nilai pixel dinormalisasi ke rentang **0–1** dengan membagi nilai pixel dengan **255**.

3. **Augmentasi Data**  
   Dilakukan augmentasi data untuk meningkatkan variasi data latih, meliputi:
   - Rotasi
   - Zoom
   - Flip horizontal

4. **Split Dataset**  
   Dataset dibagi menjadi:
   - Training
   - Validation
   - Testing

5. **Batching dan Ekspansi Dimensi**  
   Citra diubah ke dalam bentuk **batch** dan diekspansi dimensinya agar dapat diproses oleh model **TensorFlow/Keras**.



Preprocessing disesuaikan dengan karakteristik masing-masing model:

Model	Preprocessing
- CNN Scratch	Normalisasi rescale=1./255
- EfficientNetB0	preprocess_input (ImageNet)
- MobileNetV2	preprocess_input + data augmentation

Data augmentation hanya diterapkan pada training set untuk MobileNetV2.


-----------------------------------------------------------------------------
## Model yang Digunakan

1️. CNN Scratch

Dibangun dari nol menggunakan layer Conv2D dan MaxPooling

Cocok sebagai baseline model

Output layer menyesuaikan jumlah kelas dataset

2️. EfficientNetB0

Menggunakan pretrained weight ImageNet

Dua tahap training:

Training head classifier

Fine-tuning layer terakhir

Lebih efisien dengan parameter lebih sedikit

3️. MobileNetV2

Lightweight model berbasis depthwise separable convolution

Menggunakan data augmentation

Fine-tuning sebagian layer akhir

----------------------------------------------------------------
##  Training Model

Training dilakukan melalui folder training/.
```
Jalankan training:
python training/train_cnn.py
python training/train_efficientnet.py
python training/train_mobilenet.py
```

Model hasil training akan otomatis tersimpan di folder models/.


----------------------------------------------------------------------------
##  Evaluasi Model

Evaluasi dilakukan menggunakan data test yang tidak dilibatkan dalam training.

Metrik evaluasi:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

```
Jalankan evaluasi:
python evaluation/eval_cnn.py
python evaluation/eval_efficientnet.py
python evaluation/eval_mobilenet.py
```

Evaluasi digunakan untuk membandingkan performa antar model secara objektif.


----------------------------------------------------------------------------
# 📊 Perbandingan Model

Hasil evaluasi dari ketiga model dapat dibandingkan untuk melihat:

Model dengan akurasi tertinggi

Keseimbangan precision dan recall

Dampak transfer learning terhadap performa

Model pretrained (EfficientNetB0 dan MobileNetV2) umumnya memberikan performa lebih baik dibanding CNN scratch.



### CNN
=== Classification Report (CNN Scratch) ===

              precision    recall  f1-score   support

       cello       0.99      0.99      0.99       346
    clarinet       0.90      0.99      0.94       147
        drum       0.90      0.38      0.54       776
        erhu       0.99      0.99      0.99       254
       flute       0.66      0.97      0.78       209
      guitar       0.92      0.99      0.96       285
       piano       0.58      0.21      0.31       387

 | Metric | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Accuracy | - | - | **0.80** | 4413 |
| Macro Avg | **0.82** | **0.86** | **0.81** | 4413 |
| Weighted Avg | **0.83** | **0.80** | **0.78** | 4413 |


<img width="800" height="600" alt="CNN" src="https://github.com/user-attachments/assets/65b13b8e-27c1-4d2f-abec-c5d03f8e1e27" />


------------------------------------------------------------------------------
### EfficientNet
=== Classification Report (EfficientNetB0) ===

              precision    recall  f1-score   support

       cello     0.9425    0.9942    0.9677       346
    clarinet     0.9470    0.9728    0.9597       147
        drum     1.0000    0.0696    0.1301       776
        erhu     0.4078    0.9921    0.5780       254
       flute     0.8957    0.9856    0.9385       209
      guitar     0.9895    0.9965    0.9930       285
       piano     0.9897    0.4987    0.6632       387

  | Metric | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Accuracy | - | - | **0.7843** | 4413 |
| Macro Avg | **0.8502** | **0.8709** | **0.7986** | 4413 |
| Weighted Avg | **0.8797** | **0.7843** | **0.7395** | 4413 |

<img width="1280" height="692" alt="EFFICIENTNET" src="https://github.com/user-attachments/assets/37c093c7-6ae9-43b0-84a2-7fe4b18d4f62" />



-----------------------------------------------------------------------------------------------------------------------
### MobileNet
=== Classification Report (MobileNetV2) ===

              precision    recall  f1-score   support

       cello     0.9886    1.0000    0.9943       346
    clarinet     1.0000    1.0000    1.0000       147
        drum     0.9971    0.8969    0.9444       776
        erhu     1.0000    1.0000    1.0000       254
       flute     1.0000    0.9761    0.9879       209
      guitar     1.0000    1.0000    1.0000       285
       piano     0.8280    0.9948    0.9038       387

| Metric | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Accuracy | - | - | **0.9803** | 4413 |
| Macro Avg | **0.9840** | **0.9890** | **0.9856** | 4413 |
| Weighted Avg | **0.9833** | **0.9803** | **0.9806** | 4413 |


<img width="1280" height="692" alt="MOBILENET" src="https://github.com/user-attachments/assets/b960cdd9-5879-4ec3-993e-23fae600cb2d" />



----------------------------------------------------------------------------
## Kesimpulan Perbandingan Model
📊 Tabel Perbandingan Kinerja Model


🔹 Performa Keseluruhan
| Model              | Accuracy | Macro Precision | Macro Recall | Macro F1-score | Weighted F1-score |
| ------------------ | -------- | --------------- | ------------ | -------------- | ----------------- |
| **CNN Scratch**    | 0.80     | 0.82            | 0.86         | 0.81           | 0.78              |
| **EfficientNetB0** | 0.78     | 0.85            | 0.87         | 0.80           | 0.74              |
| **MobileNetV2**    | **0.98** | **0.98**        | **0.99**     | **0.99**       | **0.98**          |

🔹 Analisis Singkat per Model
| Model              | Kelebihan                                                            | Kekurangan                                                                |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **CNN Scratch**    | Arsitektur sederhana, mudah dipahami, cocok sebagai baseline         | Performa rendah pada kelas kompleks seperti *drum* dan *piano*            |
| **EfficientNetB0** | Precision tinggi pada sebagian besar kelas, efisien secara parameter | Recall sangat rendah pada kelas *drum*, menunjukkan prediksi tidak stabil |
| **MobileNetV2**    | Performa paling konsisten dan unggul di seluruh metrik evaluasi      | Membutuhkan resource komputasi lebih besar dibanding CNN scratch          |

🔹 Contoh Perbandingan Kelas Sulit
| Kelas     | CNN F1-score | EfficientNetB0 F1-score | MobileNetV2 F1-score |
| --------- | ------------ | ----------------------- | -------------------- |
| Drum      | 0.54         | 0.13                    | **0.94**             |
| Piano     | 0.31         | 0.66                    | **0.90**             |
| Xylophone | 0.66         | 0.58                    | **1.00**             |

----------------------------------------------------------------------------------------------------
### Kesimpulan Akhir

| Nama Model          | Akurasi | Hasil Analisis |
|---------------------|---------|----------------|
| CNN Scratch         | 0.80    | Model baseline tanpa pretrained. Mampu mengenali beberapa kelas dengan baik, namun performa menurun pada kelas dengan kemiripan visual tinggi seperti drum dan piano. Cocok sebagai pembanding awal. |
| EfficientNetB0      | 0.78    | Model transfer learning dengan fitur kuat, namun menunjukkan ketidakseimbangan recall pada beberapa kelas (misalnya drum dan xylophone). Performa cukup baik tetapi belum optimal pada dataset ini. |
| MobileNetV2         | 0.98    | Model terbaik dengan akurasi tertinggi dan performa stabil di hampir semua kelas. Sangat efektif, efisien, dan cocok untuk deployment aplikasi berbasis web seperti Streamlit. |

----------------------------------------------------------------------------
Kesimpulan Akhir

MobileNetV2 adalah model terbaik dengan akurasi 98% dan performa sangat konsisten di seluruh kelas.

CNN Scratch cukup baik sebagai baseline, namun terbatas untuk dataset multi-kelas kompleks.

EfficientNetB0 menunjukkan precision tinggi tetapi recall tidak stabil pada kelas tertentu, menandakan kemungkinan overfitting atau ketidakseimbangan fitur.

📌 Dari proyek yang di lakukan model terbaik untuk deployment & aplikasi nyata: MobileNetV2


----------------------------------------------------------------------------------
## 🌐 Aplikasi Streamlit

### Tampilan Dashboard Sistem

Aplikasi ini menyediakan dua mode tampilan utama, yaitu Single Model dan Bandingkan Semua Model, untuk memudahkan analisis dan perbandingan performa model klasifikasi citra alat musik.

🔹 Mode Single Model

Pada mode Single Model, pengguna dapat memilih satu model saja untuk melakukan prediksi terhadap gambar input.

Fitur pada Mode Single Model:

Upload satu gambar alat musik sebagai input

Pemilihan model:

CNN Base (Non-Pretrained)

MobileNetV2 (Transfer Learning)

EfficientNetB0 (Transfer Learning)

Hasil prediksi ditampilkan secara fokus, meliputi:

Nama kelas hasil klasifikasi

Nilai confidence (tingkat keyakinan model)

Grafik batang distribusi probabilitas seluruh kelas

Tujuan Mode Single Model:

Mengamati perilaku dan karakteristik masing-masing model secara individual

Mengevaluasi kestabilan prediksi tanpa gangguan model lain

Cocok digunakan untuk analisis mendalam (model inspection)
<img width="956" height="507" alt="Screenshot 2025-12-26 003429" src="https://github.com/user-attachments/assets/be24b49e-8ded-445a-9d32-774fc16fb899" />



🔹 Mode Bandingkan Semua Model

Mode Bandingkan Semua Model memungkinkan pengguna melihat hasil prediksi dari seluruh model secara bersamaan menggunakan satu gambar input yang sama.

Informasi yang Ditampilkan:

Gambar input yang sama untuk semua model

Hasil prediksi tiap model, yaitu:

CNN Base

MobileNetV2

EfficientNetB0

Confidence score masing-masing model

Visualisasi probabilitas kelas dalam bentuk grafik batang untuk setiap model

Analisis Contoh Hasil:

CNN Base dan MobileNetV2 menghasilkan prediksi kelas guitar dengan confidence yang sangat tinggi

EfficientNetB0 menghasilkan prediksi berbeda dengan confidence rendah, yang menunjukkan kemungkinan kesalahan klasifikasi atau model belum optimal pada sampel tertentu

Tujuan Mode Bandingkan Semua Model:

Membandingkan performa antar model secara langsung (side-by-side comparison)

Mengidentifikasi model dengan akurasi dan kepercayaan prediksi terbaik

Mendukung pengambilan keputusan dalam pemilihan model terbaik
<img width="959" height="506" alt="Screenshot 2025-12-26 003458" src="https://github.com/user-attachments/assets/571b78ba-8b85-4941-ada3-e6792ad4ebce" />



🔹 Manfaat Dashboard Secara Keseluruhan

Dashboard ini dirancang untuk:

Menyediakan visualisasi hasil prediksi yang interaktif

Memudahkan evaluasi dan interpretasi hasil klasifikasi citra alat musik

Mendukung eksperimen perbandingan model CNN dan Transfer Learning

Menjadi alat bantu analisis dalam penelitian atau tugas akademik berbasis Deep Learning

----------------------------------------------------------------------------
## ⚙️ Instalasi dan Menjalankan Aplikasi
1. Clone Repository
```
git clone https://github.com/tantri17/UAP_tantri-240
```
Beberapa data tambahan harus diunduh melalui link yang tersedia, silakan baca instruksi dengan teliti.

2. Masuk ke Folder Project
```
cd ML_STREAMLIT_DASHBOARD
```
3. Buat Virtual Environment
```
python -m venv .venv
```
4. Aktivasi Environment

```
Windows
.venv\Scripts\activate

macOS / Linux
source .venv/bin/activate
```
5. Instalasi Dependensi
```
pip install streamlit tensorflow numpy pandas pillow
```
6. Jalankan Aplikasi
```
streamlit run app.py
```
7. Akses Dashboard

```
Buka browser dan akses:
Local URL: http://localhost:8501

Network URL: http://192.168.0.30:8501
```










