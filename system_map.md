# Ultrason System Map

## Context Dosyaları Rehberi

| Görev Türü | Oku | Açıklama |
|------------|-----|----------|
| Proje yapısı, tech stack, dosya haritası | **system_map.md** (bu dosya) | Projenin statik anatomisi |
| Günlük değişiklikler, ne yapıldı | **Developments.md** | Kronolojik changelog |
| Literatür taraması, AI mimari değerlendirmesi | **Literatur_Raporu.pdf** | 30+ çalışmalık literatür incelemesi |
| Meme dokusu akustik/termal parametreleri | **Gorsel1.jpeg** | Tablo 8 (yağ/fibro/tümör ρ, c, α, ω) |
| k-Wave 3D Arşimet transdüser konfigürasyonu | **Gorsel2.jpeg** | Mevcut akustik simülasyon kurulumu |
| Session notları, linkler | **Claud.txt** | Önceki konuşma referansları, SPIE/MDPI linkleri |
| Kurulum ve kullanım | **README.md** | Yol haritası + kurulum + OpenBreastUS indirme |

> Henüz arşiv klasörü yok. İleride `docs/archive/` altına haftalık snapshot alınacak.

---

## Project Overview

**Ultrason**, yumuşak meme dokusunda ultrason basınç dağılımını yapay zeka ile tahmin etmek için bir **Fourier Neural Operator (FNO)** vekil modeli geliştiren bir AI/ML araştırma projesidir. Nihai hedef, HIFU (High-Intensity Focused Ultrasound) tümör ablasyon tedavisinin gerçek-zamanlı planlamasını mümkün kılmak — k-Wave gibi geleneksel çözücülerin saatler süren hesaplamalarını milisaniyelere indirmek.

**Tip:** AI/ML araştırma projesi (Python)
**Araştırma sorusu:** N hasta üzerinde eğitilen bir operatörün, görmediği (N+1). hastada tekrardan eğitmeden doğru basınç alanı tahmini üretmesi — özellikle literatürde boş bırakılmış **yumuşak doku heterojenliği** için.

---

## Related Documentation

**Location:** `c:\Users\gokss\OneDrive\Masaüstü\Ultrason\`

| Dosya | Açıklama |
|---|---|
| [README.md](README.md) | Kurulum, quick start, veri setini indirme adımları |
| [Literatur_Raporu.pdf](Literatur_Raporu.pdf) | Kadir Göksel Gündüz, PINN/CNN/NO/Transformer literatür taraması |
| [Claud.txt](Claud.txt) | Önceki konuşma linkleri, session notları |
| [Gorsel1.jpeg](Gorsel1.jpeg) | Meme dokusu Tablo 8 — fiziksel parametreler |
| [Gorsel2.jpeg](Gorsel2.jpeg) | 3D Arşimet transdüser + odak bölgesi simülasyonu |

---

## Entry Points — AI/ML Pipeline

Bu proje CLI script'leri üzerinden çalışır. Her script bağımsız bir pipeline aşamasıdır.

**Kol 1 — OpenBreastUS forward (doku → basınç)**

| Stage | Script | Amaç |
|---|---|---|
| Sentetik smoke test | `scripts/run_single_test.py` | Sentetik fantom + k-Wave + görselleştirme — pipeline doğrulama |
| Gerçek fantom testi | `scripts/run_openbreastus_test.py` | OpenBreastUS fantomu + native speed path + tek simülasyon |
| Dataset üretimi | `scripts/generate_dataset.py` | Toplu `(c, ρ, α) → p_max` çifti üretimi → HDF5 |
| FNO eğitimi | `scripts/train_fno.py` | FNO2d (neuraloperator) + log-target + LpLoss+H1Loss |
| U-Net baseline | `scripts/train_unet.py` | 2D U-Net baseline — FNO ile karşılaştırma |
| Model karşılaştırma | `scripts/compare_models.py` | FNO vs U-Net yan yana, 5-panel figür |

**Kol 2 — Eren inverse (Q → fazlar)**

| Stage | Script | Amaç |
|---|---|---|
| Veri preprocess | `scripts/preprocess_eren.py` | 5 zip stream → 1 compact HDF5 (extract etmeden) |
| Inverse eğitim | `scripts/train_eren_inverse.py` | 3D CNN + MLP → 256 faz (sin/cos) |

**Kullanım örnekleri:**
```bash
python scripts/run_single_test.py                                       # 1 sentetik sample
python scripts/run_openbreastus_test.py                                 # 1 gerçek fantom
python scripts/generate_dataset.py --mat breast_train_speed.mat \
    --n 1000 --out data/dataset_v1.h5                                   # toplu üretim
python scripts/train_fno.py --data data/dataset_v1.h5 --epochs 80 \
    --n-modes 32 --hidden-channels 64 --n-layers 5 --loss lp+h1         # FNO eğit
python scripts/train_unet.py --data data/dataset_v1.h5 --epochs 80      # U-Net baseline
```

---

## Tech Stack

### ML / Simulation Core

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| k-wave-python | 0.6.1 | Akustik dalga simülasyonu (k-space pseudo-spectral) |
| neuraloperator | 2.0.0 | FNO2d implementasyonu, LpLoss, Trainer |
| PyTorch | ≥2.1 | Deep learning backend |
| NumPy | ≥1.24 | Sayısal işlemler |
| SciPy | 1.15.3 | `loadmat` (OpenBreastUS .mat okuma), `zoom` (resampling) |
| h5py | ≥3.9 | Dataset paketleme |
| matplotlib | ≥3.7 | Görselleştirme |

### Veri Kaynağı

| Kaynak | Versiyon | Amaç |
|---|---|---|
| OpenBreastUS | arXiv:2507.15035 (Jul 2025) | 8000 anatomik meme fantomu, CC-BY 4.0 |
| UWCEM (opsiyonel) | – | 9 MRI-türevli meme fantomu (fallback) |

### Runtime

- **Paket yöneticisi:** pip (requirements.txt)
- **Platform:** Windows 11 (bash shell üzerinden)
- **GPU:** CPU-only şu anda (~4.5 s/simülasyon, 256×256); k-wave-python GPU binary ileride eklenecek

### Mimari Diyagram

```
[OpenBreastUS .mat]
       ↓
[phantom.load_openbreastus_speed]         (continuous c(x,y))
       ↓
[phantom.insert_tumor_speed]              (tümör c=1550 m/s)
       ↓
[simulate.speed_to_density/alpha]         (ρ, α sürekli türetme)
       ↓
[simulate.run_focused_sim_from_speed]     (k-wave-python 2D)
       ↓
[outputs: npz + PNG preview]
       ↓
[scripts.generate_dataset → HDF5]
       ↓
[dataset.PressureFieldDataset]
       ↓
[neuraloperator.FNO2d train]
       ↓
[outputs/fno_run: best.pt + loss_curve + test_sample]
```

---

## Data Layer

Bu proje **stateless/dosya-tabanlı** bir AI/ML projesidir — veritabanı yoktur. Veri akışı aşağıdaki gibidir:

### Girdi Verisi (Input)

**OpenBreastUS .mat dosyaları:**

| Dosya | Key | Shape | Dtype | Değer aralığı | Boyut |
|---|---|---|---|---|---|
| `breast_test_speed.mat` | `breast_test` | `(800, 480, 480)` | float32 | 1403–1597 m/s | 195 MB |
| `breast_train_speed.mat` | `breast_train` | `(~7200, 480, 480)` | float32 | ~aynı | 1.69 GB |

Her "fantom" zaten 2D bir 480×480 meme kesiti ses hızı haritasıdır. Dilim seçmeye gerek yok.

### Simülasyon Çıktısı (Intermediate)

Her k-Wave koşusu şu sözlüğü üretir:

| Alan | Shape | Açıklama |
|---|---|---|
| `c` | (H, W) float32 | Ses hızı haritası (m/s) |
| `rho` | (H, W) float32 | Yoğunluk haritası (kg/m³) |
| `alpha` | (H, W) float32 | Zayıflama katsayısı (dB/(MHz^y·cm)) |
| `p_max` | (H, W) float32 | Tepe basınç alanı (Pa) — **FNO'nun hedefi** |
| `focus_yx` | (2,) int32 | Hedef koordinatı (axial, lateral) |
| `source_mask` | (H, W) bool | Kaynak transdüser konum maskesi |

### Training Dataset (HDF5)

`scripts/generate_dataset.py` aşağıdaki layout'u yazar:

| Dataset | Shape | Dtype | İçerik |
|---|---|---|---|
| `inputs` | `(N, 3, H, W)` | float32 | (c, ρ, α) kanalları |
| `targets` | `(N, 1, H, W)` | float32 | Tepe basınç |
| `labels` | `(N, H, W)` | uint8 | Etiket haritası (opsiyonel, native path'te kullanılmaz) |
| `focus` | `(N, 2)` | int32 | Hedef koordinatı |

Attrs: `source_f0_Hz`, `dx_m`, `source_amp_Pa`.

### Model I/O

**FNO2d:**
- **Input:** `(B, 3, H, W)` — normalize edilmiş (c, ρ, α)
- **Output:** `(B, 1, H, W)` — normalize edilmiş peak pressure (`y_scale = max |p|`)
- **Loss:** `neuralop.losses.LpLoss(d=2, p=2)`

---

## Authentication & Security

Yok. Lokal araştırma projesi, auth gerekmez. OpenBreastUS verisi CC-BY 4.0 lisanslı, public.

---

## Project Structure

**Root:** `c:\Users\gokss\OneDrive\Masaüstü\Ultrason\`

```
Ultrason/
├── README.md                              # Kurulum, quick start, yol haritası
├── requirements.txt                       # Python bağımlılıkları
├── system_map.md                          # Projenin statik anatomisi (bu dosya)
├── Developments.md                        # Kronolojik changelog
├── Claud.txt                              # Session notları, önceki konuşma linkleri
├── Literatur_Raporu.pdf                   # 30+ çalışmalık AI literatür taraması
├── Gorsel1.jpeg                           # Tablo 8 — meme dokusu fiziksel parametreleri
├── Gorsel2.jpeg                           # 3D Arşimet transdüser + odak simülasyonu
│
├── data/                                  # Veri dosyaları (gitignore'lanmalı)
│   ├── breast_test_speed.mat              #   OpenBreastUS 800 test fantomu (195 MB)
│   └── breast_train_speed.mat             #   OpenBreastUS ~7200 train fantomu (1.69 GB)
│
├── src/                                   # Kütüphane kodu
│   ├── tissue_properties.py               #   Etiket → (c, ρ, α) mapping, Tablo 8 değerleri
│   ├── phantom.py                         #   Sentetik + OpenBreastUS loader + tümör insert
│   ├── simulate.py                        #   k-wave-python 2D focused sim (label + native)
│   ├── dataset.py                         #   HDF5 Dataset, log-target, RAM cache
│   ├── unet.py                            #   2D U-Net baseline mimarisi
│   ├── eren_model.py                      #   Kol 2: PhaseInverseNet (3D CNN + MLP + sin/cos)
│   └── eren_dataset.py                    #   Kol 2: ErenPhaseDataset (Q → sincos)
│
├── scripts/                               # CLI entry points
│   ├── run_single_test.py                 #   Kol 1: sentetik fantom smoke test
│   ├── run_openbreastus_test.py           #   Kol 1: gerçek fantom NATIVE test
│   ├── generate_dataset.py                #   Kol 1: toplu dataset üreticisi → HDF5
│   ├── train_fno.py                       #   Kol 1: FNO2d eğitim
│   ├── train_unet.py                      #   Kol 1: U-Net baseline eğitim
│   ├── compare_models.py                  #   Kol 1: FNO vs U-Net 5-panel karşılaştırma
│   ├── preprocess_eren.py                 #   Kol 2: zip stream → compact HDF5
│   └── train_eren_inverse.py              #   Kol 2: 3D CNN inverse eğitim
│
└── outputs/                               # Simülasyon ve eğitim çıktıları
    ├── sample_0000.npz                    #   Sentetik smoke test ham verisi
    ├── sample_0000.png                    #   Sentetik test 3-panel önizleme
    ├── obus_phantom_0000.npz              #   Quantized path tek fantom
    ├── obus_phantom_0000.png              #   Quantized path 4-panel
    ├── obus_native_0000.npz               #   NATIVE path tek fantom
    └── obus_native_0000.png               #   NATIVE path 4-panel (final)
```

---

## Key Components / Modules

### Src Modülleri

| Module | File | Purpose |
|---|---|---|
| `tissue_properties.labels_to_maps` | `src/tissue_properties.py` | Discrete label map → (c, ρ, α) tuple |
| `phantom.make_synthetic_phantom` | `src/phantom.py` | Prosedurel 2D meme fantomu (debug) |
| `phantom.load_openbreastus_speedmaps` | `src/phantom.py` | `.mat` → stack (N, 480, 480) float32 |
| `phantom.load_openbreastus_speed` | `src/phantom.py` | Tek fantomu çekip opsiyonel resize |
| `phantom.load_openbreastus_phantom` | `src/phantom.py` | Quantized versiyon (LABEL path) |
| `phantom.insert_tumor` | `src/phantom.py` | Discrete label map'e tümör yerleştir |
| `phantom.insert_tumor_speed` | `src/phantom.py` | Continuous speed map'e tümör (c=1550) |
| `simulate.speed_to_density` | `src/simulate.py` | ρ = 0.85·c − 282 (Mast 2000 türevi) |
| `simulate.speed_to_alpha` | `src/simulate.py` | α(c) — fibroglandüler zirveli Gauss karışımı |
| `simulate._run_with_medium` | `src/simulate.py` | Ortak k-Wave driver, her iki path buraya gelir |
| `simulate.run_focused_sim` | `src/simulate.py` | LABEL path — tissue_properties ile quantize |
| `simulate.run_focused_sim_from_speed` | `src/simulate.py` | **NATIVE path** — sürekli heterojenliği korur (ana yol) |
| `simulate.SimConfig` | `src/simulate.py` | dx, f0, t_end, aperture, source offset ayarları |
| `dataset.PressureFieldDataset` | `src/dataset.py` | HDF5 → tensor, per-channel normalize + log-space target scaling |
| `dataset.PressureFieldDataset.denormalize_target` | `src/dataset.py` | Normalize edilmiş prediction → Pa birimine dönüş |
| `unet.UNet2d` | `src/unet.py` | 4-level 2D U-Net encoder/decoder, in=3 out=1 |

### Script Pipeline Adımları

| Stage | Script | Purpose |
|---|---|---|
| Smoke test (synth) | `scripts/run_single_test.py` | Sentetik fantom + tek sim + 3-panel PNG |
| Smoke test (real) | `scripts/run_openbreastus_test.py` | OpenBreastUS idx=0 + native sim + 4-panel PNG |
| Dataset build | `scripts/generate_dataset.py` | N sample loop → `data/dataset_v{N}.h5` |
| FNO train | `scripts/train_fno.py` | FNO2d + lp+h1 loss + log-target, best checkpoint, loss curve, test sample figure |
| U-Net train | `scripts/train_unet.py` | Aynı dataset/loss arayüzü, U-Net baseline, FNO ile karşılaştırmalı |

---

## Mevcut Sonuçlar

**Kol 1 — OpenBreastUS 1000 sample (FNO vs U-Net karşılaştırması):**

| Model | Params | Test LpLoss (train script) | Test LpLoss (compare, sample-wise) |
|---|---|---|---|
| **FNO2d** | 5.16M | 0.383 | **0.097** |
| U-Net2d | 7.85M | 1.042 | 0.264 |

FNO, U-Net'e göre **2.7× daha iyi** (compare_models.py sample-wise ölçümü). 200 sample smoke test'inde fark 1.5×'tı — data scaling ile avantaj büyüyor.

**Kol 2 — Eren inverse (100 sample):**
- Smoke test başarısız: val phase error 90.9° (rastgele). 100 sample yetersiz + ill-posed formülasyon. Eren 3000 sample'ı tamamlayınca tekrar + target_pt conditioning denenecek.

---

## Önemli Kararlar

1. **NATIVE path birincil, LABEL path yedek.** OpenBreastUS sürekli ses hızı haritasını quantize etmek merdiven artifakları ve yanlış refraksiyon yaratıyor. Native path tüm heterojenliği korur ve makale için güçlü argüman sağlar ("OpenBreastUS fantomlarını HIFU rejimine orijinal çözünürlükte taşıyan ilk çalışma").
2. **2D ile başla, 3D sonra.** Literatürdeki başarılı modellerin (TUSNet, Kumar 2024) çoğu 2D dilimlerde çalışıyor. 3D'ye sonra geçilir.
3. **FNO birincil, DeepONet ikincil.** `neuraloperator` kütüphanesi hazır; image-to-image form doğal; çözünürlükten bağımsız. DeepONet ikinci iterasyonda gelecek.
4. **k-Wave ground truth sayılır.** Literatürdeki tüm vekil model çalışmaları (Kumar, TUSNet, Stanziola) sentetik k-Wave çıktısını doğru kabul ediyor — gerçek hidrofon ölçümü değil.
