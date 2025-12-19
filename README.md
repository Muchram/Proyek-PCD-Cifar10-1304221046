# PROYEK AKHIR PENGOLAHAN CITRA DIGITAL  
## Klasifikasi Gambar CIFAR-10 Menggunakan Preprocessing + MobileNetV2 Feature Extraction + SVM (Streamlit App)

---

## Identitas
- **Nama**: Yoga Muchram Anarqi
- **NIM**: 1304221046
- **Kelas**: IF-46-01.1PJJ
- **Mata Kuliah**: Pengolahan Citra Digital
- **Program Studi / Fakultas**: Teknik Informatika PJJ
- **Universitas**: Telkom University

---

## 1. Deskripsi Singkat Proyek
Proyek ini mengimplementasikan pipeline pengolahan citra dan klasifikasi untuk memenuhi tugas akhir mata kuliah **Pengolahan Citra Digital**.  
Tahapan yang digunakan:
1) **Pre-processing**: reduksi noise dan peningkatan kualitas citra (enhancement).  
2) **Feature Extraction**: ekstraksi fitur menggunakan deep learning pretrained (**MobileNetV2**).  
3) **Classification**: klasifikasi menggunakan **SVM (LinearSVC)**.  
4) **Post-processing / Evaluation**: evaluasi performa dengan metrik **Accuracy, Precision, Recall, F1-score**, serta **Confusion Matrix**, dan fine-tuning parameter SVM.

Dataset: **CIFAR-10** (10 kelas):  
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

---

## 2. Pemetaan ke Komponen Tugas (Sesuai Instruksi Proyek)
### 2.1 Pre-processing
- Noise reduction: **Gaussian Filter**, **Median Filter**
- Image enhancement: **CLAHE (Histogram Equalization adaptif)**, **Contrast Stretching**

### 2.2 Feature Extraction
- **MobileNetV2 pretrained (ImageNet)** sebagai deep feature extractor (transfer learning).

### 2.3 Classification
- **SVM (LinearSVC)** dari scikit-learn untuk klasifikasi berdasarkan fitur hasil ekstraksi.

### 2.4 Post-processing
- Evaluasi: **Accuracy, Precision, Recall, F1-score**, serta **Confusion Matrix**
- Fine-tuning: **GridSearch parameter C** pada SVM (dijalankan di notebook/Colab)

---

## 3. Struktur Folder / File
File utama dalam repository:
- `app.py` -> aplikasi web berbasis Streamlit untuk demo klasifikasi
- `requirements.txt` -> dependency Python untuk menjalankan aplikasi
- `.gitignore` -> mencegah file besar/tidak perlu (misal `.venv`) ikut ter-upload
- `svm_cifar10_mobilenet.joblib` -> model SVM hasil training (digunakan oleh aplikasi)
- `preprocess_config.joblib` -> konfigurasi preprocessing (digunakan oleh aplikasi)

Catatan:
- Folder `.venv/` **tidak diupload** ke GitHub karena merupakan environment lokal dan ukurannya besar. Dependency digantikan oleh `requirements.txt`.

---

## 4. Cara Menjalankan Aplikasi (Windows) — Step-by-Step
### 4.1 Masuk ke folder proyek
Buka PowerShell, lalu:
```powershell
cd "D:\Telkom University\SEMESTER 7\PENGOLAHAN CITRA DIGITAL\cifar_app" (Semisalkan Bila di Direktori Komputer Saya)

### 4.2 Aktifkan virtual environment
Dengan kode : .\.venv\Scripts\activate
(Jika berhasil, prompt akan berubah menjadi ada (.venv))


### 4.3 Install dependency
Dengan kode : pip install -r requirements.txt


### 4.4 Jalankan aplikasi Streamlit
Dengan kode : python -m streamlit run app.py

Buka URL yang muncul di terminal biasanya seperti:
http://localhost:8501

---

## 5. Cara Menggunakan Aplikasi
### 1. Upload gambar .jpg atau .png.

### 2. Aplikasi akan menampilkan:

```gambar Original

```gambar After Preprocessing

```hasil prediksi kelas (Top-1) serta Top-3 score.

***Saran gambar untuk uji coba (lebih sesuai dengan CIFAR-10):
kucing, anjing, burung, katak, kuda, rusa, mobil, truk, pesawat, kapal

***Catatan: Model dilatih pada CIFAR-10 (ukuran 32×32). Jika gambar upload berasal dari kamera (resolusi besar), prediksi bisa kurang akurat