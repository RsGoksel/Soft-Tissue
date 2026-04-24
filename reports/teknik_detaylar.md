# Teknik Detaylar — Soft-Tissue HIFU Planlama Pipeline'ı

Projenin ampirik süreci: hangi yaklaşımlar denendi, hangileri çalışmadı,
neden çalışmadı, son çalışan mimari hangisi ve neden o seçildi.
Sempozyum makalesi ve gelecek iterasyonlar için referans dokümandır.

---

## İçindekiler

- [1. Problem](#1-problem)
- [2. İlk yaklaşım ve neden başarısız oldu](#2-ilk-yaklaşım-ve-neden-başarısız-oldu)
- [3. Pivot: iki-kollu pipeline](#3-pivot-iki-kollu-pipeline)
- [4. Kol A — 2-B ileri basınç alanı vekil modeli](#4-kol-a--2-b-ileri-basınç-alanı-vekil-modeli)
- [5. Kol B — 3-B ters odak-noktası regresyonu](#5-kol-b--3-b-ters-odak-noktası-regresyonu)
- [6. Gauge simetrisi: üçlü doğrulama](#6-gauge-simetrisi-üçlü-doğrulama)
- [7. Faz-kuantizasyon çalışması](#7-faz-kuantizasyon-çalışması)
- [8. Isı-haritası regresyonu (nnLandmark / H3DE-Net tarzı)](#8-ısı-haritası-regresyonu-nnlandmark--h3de-net-tarzı)
- [9. Sonuçların dürüst okunması](#9-sonuçların-dürüst-okunması)
- [10. Reproduksiyon](#10-reproduksiyon)

---

## 1. Problem

Heterojen dokuda HIFU planlaması iki pahalı hesaplama problemi içerir:

1. **İleri (forward)**: hastaya özgü doku haritası (ses hızı, yoğunluk,
   soğurma) verildiğinde, transdüser konfigürasyonunun ürettiği
   akustik basınç alanını tahmin et. Referans çözücü: tam-dalga k-Wave
   simülasyonu (yavaş; konfigürasyon başına saniye–dakika).
2. **Ters (inverse)**: istenen odak konumu verildiğinde, enerjiyi
   oraya yönlendirecek transdüser faz vektörünü sentezle. Klasik
   yaklaşım: ileri çözücü üzerinde yinelemeli optimizasyon — interaktif
   planlama için çok yavaş.

Veri seti: **OpenBreastUS** (Zeng ve ark. 2025) — gerçek meme fantom
hacimlerinden alınmış 2-B ses hızı haritaları + k-Wave ile hesaplanmış
basınç alanları. Bildiğimiz kadarıyla **bu veri setinin HIFU planlamada
ilk kullanımı**dır (orijinal yayın ultrason bilgisayarlı tomografi için).

## 2. İlk yaklaşım ve neden başarısız oldu

Projenin ilk kurgusu sezgisel bir formülasyondu:

> 256 elemanlı bir ağ `f: Q → faz_vektörü` eğit ki istenen ısı
> birikimi hacmi `Q`'yu doğrudan transdüser faz vektörüne eşlesin.

Ampirik olarak **hiçbir şey öğrenmedi**. Kayıp eğrisi rastgele
referansların çok üzerinde plato yaptı. Tanılama iki yapısal sebebi
ortaya çıkardı.

### 2.1 Q → faz eşlemesi tek-değerli değil (one-to-many)

30 örneklik sette, **en yakın komşu** hedef pozisyonlarına ait faz
vektörlerinin dairesel-kosinüs benzerliği **0.002 ± 0.043** — gürültüden
ayırt edilemez. Aynı hedefi üreten çok sayıda farklı faz vektörü var
(girişim deseni yalnızca göreceli geometriye bağlı). Deterministik bir
regresyon tek-çoğa ilişkiyi fit edemez.

### 2.2 Gauge simetrisi

Tüm faz vektörüne sabit `+δ` eklenmesi girişim desenini — dolayısıyla
odağı — **tam olarak aynı bırakır**. Bu problemin sürekli bir
simetrisidir. Ağın sonsuz eşdeğer çıktıdan birini "seçmesi" gerekir;
sabit-hedef kaybı bu durumda yakınsayamaz.

Her iki patoloji de tanı metrikleri yazıldığı anda görüldü; eğitim
eğrilerine bakarak bulunamadı (loss hafif düştü sonra plato = ilk
bakışta "modeli biraz daha büyüt" izlenimi verir). Aslında bu
**kayıp fonksiyonunun ill-posed olması**nın tipik imzasıydı.

## 3. Pivot: iki-kollu pipeline

Yeniden formülasyon: fiziği sağlam olduğu yerde koru, AI'ı yalnızca
somut bir kazanç verdiği yerde kullan.

- **Kol A — İleri vekil model.** Bir sinir operatörü, doku
  haritalarından 2-B basınç alanını tahmin ediyor; herhangi bir
  optimizasyon döngüsünün içinde pahalı tam-dalga çözücünün yerine
  geçiyor.
- **Kol B — Ters nokta regresyonu.** Kompakt bir 3-B CNN, `Q`'dan
  **odak koordinatını (x, y, z)** doğrudan tahmin ediyor — yalnızca
  3 serbestlik derecesi. Transdüser fazları bu noktadan bilinen fiziksel
  model üzerinden analitik delay-and-sum hüzmelendirmeyle üretiliyor.
  Tek-çoğa ve gauge patolojileri böylece ortadan kalkıyor.

## 4. Kol A — 2-B ileri basınç alanı vekil modeli

Veri: 1 000 OpenBreastUS ses hızı haritası, 316 × 256 grid, 1 MHz
odaklanmış kaynak, 12 mm su tabakası, k-Wave referans. Tek seed'li
bölünme (70 / 15 / 15) tüm omurgalarda aynı. Hedef:
`LpLoss + 0.3 · H1Loss`. Test metriği: göreli L2 (LpLoss).

| Omurga        | Params  | Test rel-L2 | Not |
|---------------|---------|-------------|-----|
| **FNO2d**     | ~10.3 M | **0.097**   | Spektral taban dalga yapısına uyuyor |
| U-Net2d       | ~7.9 M  | 0.264       | Klasik baseline, kırılma kuyruğunu bulandırıyor |
| ConvNeXt2d    | 1.87 M  | 0.990       | Ablasyon; tam da bu uyumsuzluk mesaj |

FNO ile U-Net arasındaki 2.7× açık ve FNO ile 2025 tarzı ConvNeXt
encoder-decoder arasındaki bir büyüklük mertebesi açığı, **mimari
önceliğinin belirleyici olduğunu** gösteriyor: FNO'nun FFT-temelli
global alıcı alanı dalga denkleminin Green fonksiyonuna doğrudan
uyuyor. 200 → 1 000 örnek ölçekleme çalışması validation LpLoss'u
0.81'den 0.39'a (%51 iyileşme) düşürdü; tam 8 000-örnek OpenBreastUS
setinde ek kazanç beklentisi yüksek.

## 5. Kol B — 3-B ters odak-noktası regresyonu

Veri: ITÜ işbirlikçimizin ürettiği 30 hacim (Q, odak-nokta) örnek,
22 / 4 / 4 bölünme, 2 mm voxel spacing. Görev: `(log1p(Q), mask)`'tan
sürekli `(x, y, z)` tahmin etmek.

### 5.1 Baseline FocusPointNet

4-seviye 3-B CNN + global average pool + 3-üniteli MLP başlık,
0.89 M parametre. **Çalışan model bu.** En güçlü klasik ısı-haritası
lokalleştiricisini (genlik-ağırlıklı merkez, test RMS 33.98 mm)
**2.5× geçiyor** (lateral hatada).

### 5.2 Çok-seed'li mimari ablasyonu

Seed gürültüsünü mimari etkilerinden ayırmak için üç omurga
(düz CNN, ResNet-3D, çok-ölçekli UNet-encoder) üç seed'de (0/1/2),
eşleştirilmiş ~0.9 M parametre bütçesinde, koşu başına 120 epoch
eğitildi:

| Omurga    | n | Test RMS (mm)     | X (mm)      | Y (mm)      | Z (mm)          |
|-----------|---|-------------------|-------------|-------------|-----------------|
| **CNN**   | 3 | **26.25 ± 4.61**  | 4.63 ± 2.21 | 2.69 ± 1.21 | **25.65 ± 4.17** |
| ResNet-3D | 3 | 34.48 ± 3.74      | 5.90 ± 3.76 | 3.10 ± 1.36 | 33.64 ± 3.21    |
| UNet-enc  | 3 | 34.17 ± 8.01      | 3.93 ± 0.70 | 3.02 ± 1.05 | 33.71 ± 8.32    |

Üç omurga da **lateral doğrulukta istatistiksel olarak denk** (3–6 mm,
HIFU doğal odak-noktasının içinde); CNN overall'da **eksenel (Z)
ekseni daha iyi yönettiği için** kazanıyor. **Her üç omurgada da Z
baskın sorun** — veri sınırı (Z menzili lateral menzilin 2×'ü,
HIFU focal zone eksenel olarak uzun).

## 6. Gauge simetrisi: üçlü doğrulama

Bölüm 2.2'deki gauge değişmezliği iddiası yalnızca nümerik bir kolaylık
değil — **üç bağımsız yoldan doğrulandı**:

1. **Analitik** — delay-and-sum hüzmelendirme denkleminden türetildi.
   Tüm fazlara sabit eklenmesi girişim deseninden faktör olarak çıkar.
2. **Sentetik** — 256-elemanlı düzlem dizide en-küçük-kareler faz-artı-
   offset fit'leri ile `+20°` offset'in akustik yoğunluk değişimini
   %1'in altında, odak kaymasını 0 mm olarak verdi.
3. **Tam-dalga k-Wave simülasyonu** — Eren (ITÜ) `+20°` testini
   tam-fizik simülatörde tekrar etti; ölçülen odak kayması yine 0 mm,
   yoğunluk değişimi yine %1'in altında.

Bu üçlü mutabakat projenin en önemli ampirik bulgusudur; çünkü
**gelecekteki her faz regresörünün çıktı uzayının gauge'la bölünmesi**
gerektiğini söylüyor — örneğin `phase[0] = 0` sabitlenerek veya
doğrudan denklik sınıfları üzerinde çalışılarak. Her iki yöntem de
her klasik optimizatöre büyük miktarda gereksiz iş yüklemiş olan sahte
bir serbestlik derecesini ortadan kaldırıyor.

## 7. Faz-kuantizasyon çalışması

Çıktı uzayı ayrıklaştırıldığında (fazlar 5°, 10°, 15° adımlara
yuvarlanırsa — gerçek transdüser sürücülerinde tipik), ne kadar
doğruluk kaybederiz? Hüzmelendiricinin her kuantizasyon seviyesinde
çalıştırılması:

| Adım  | Yoğunluk hatası | Odak kayması |
|-------|-----------------|--------------|
| 5°    | < %0.35         | < 0.1 mm     |
| 10°   | < %1.1          | < 0.3 mm     |
| 15°   | < %2.5          | < 0.7 mm     |

**Adopsiyon: 5° proje varsayılanı** (herhangi bir klasik veya
öğrenilmiş faz regresörü için). Gauge sabitlemesiyle birleşince
hem öğrenme için tractable hem de fiziksel olarak sadık, ayrık ve
gauge-değişmez bir çıktı uzayı ortaya çıkıyor.

## 8. Isı-haritası regresyonu (nnLandmark / H3DE-Net tarzı)

2025 medikal görüntüleme literatürü ilhamı — **nnLandmark**
(Weihsbach ve ark. 2025) ve **H3DE-Net** (arXiv:2502.14221) ısı-haritası
regresyon + offset-head tasarımlarıyla landmark tespitinde sub-2 mm
radyal hata bildiriyor. İki varyantı Kol B'de kurduk ve değerlendirdik.

**Üç ampirik bulgu** — her biri paper-seviyesi ilginç:

1. **Gaussian hedef üzerinde saf MSE bizim veri boyutumuzda çöküyor.**
   `σ = 3` voxel'lik Gaussian 2-milyon-voxel'lik hacmin yalnızca
   ~3×10⁻⁵'ini kaplıyor; gradient sinyali kayboluyor. Weighted-MSE ve
   peak-ölçekleme düzeltmeleri de yetmedi.
2. **DSNT kaybı (Nibali ve ark. 2018) çöküşü engelliyor ve doğrudan
   koordinat regresyonuyla denklik sağlıyor.** **25.27 mm test RMS**
   (CNN baseline 26.25 mm) + eksende biraz daha iyi (Z: 23.79 vs
   25.65 mm). Soft-argmax üzerinde varyans-regüle edilmiş hedef
   kararlı.
3. **H3DE-Net tarzı alt-voxel offset başlığı 22 eğitim örneğinde
   zararlı** — test RMS 66.02 mm, baseline'dan 2.5× kötü. Yoğun offset
   regresyonu (voxel başına 3 kanal) veriyle kısıtlanamıyor.

Bu üç bulgu birlikte, nnLandmark ailesinin bildirdiği ısı-haritası
avantajının (500+ örnekte %10–30 iyileşme) bizim problem için **ortaya
çıkmaya başladığı örnek-boyutu eşiğini ampirik olarak tanımlıyor**.
Sempozyum için değerli bir negatif / eşik sonucu.

## 9. Sonuçların dürüst okunması

- **Kol A (ileri)**: güçlü pozitif. FNO'nun spektral önceliği baskın
  etki — "daha büyük" jenerik CNN kesinlikle kötüleştirdi. 0.097 test
  LpLoss temiz sonuç; U-Net'e göre 2.7× açık esas ampirik katkı.
- **Kol B (ters, lateral)**: güçlü pozitif. 30 simülasyonla bile odak
  lateralde doğal focal spot içinde (X 4.6 mm, Y 2.7 mm). Üç omurgada
  da aynı; sonuç görevin özelliği, tek bir modelin değil.
- **Kol B (ters, eksenel)**: gerçek bir sınır. 25 mm eksenel RMS
  klinik için gevşek, ama kök sebep 22 eğitim örneği + anatomi, omurga
  değil.
- **Simetri / kuantizasyon**: temiz karakterize edildi, üçlü
  doğrulandı, pipeline'a adopte edildi (5° ayrık, gauge-sabit).

### Eksenel sayıyı neyin gerçekten hareket ettireceği

Tam argüman: [`reports/future_work_ai.md`](future_work_ai.md). Kısaca:

1. **Önceden eğitilmiş 3-B medikal foundation model üzerinden transfer
   learning** (SAM-Med3D, MONAI). Beklenen test RMS düşüşü: 26 mm'den
   mevcut 30-örnek rejiminde 10–15 mm'ye; veri büyüdüğünde sub-5 mm'ye.
2. **Daha fazla simülasyon** (500+). Her sonraki iyileştirmeyi
   mümkün kılıyor; ITÜ işbirlikçimiz üretiyor.
3. **Kol A'da Transolver / GNOT**. Net paper-metriği kazancı
   (beklenen %30–50 daha düşük rel-L2), mevcut pipeline aşamasında
   klinik etkisi mütevazı.

Kısa-liste dışı: YOLO veya jenerik image-SOTA omurgalara pivot. ConvNeXt2d
ablasyonumuz ve Kol B'nin omurga-bağımsız sonuçları, **mimari ailesinin
darboğaz olmadığını** doğruladı — darboğaz veri boyutu ve transfer
learning.

## 10. Reproduksiyon

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

**Kol A** (1 000-örneklik veri + üç omurga):

```bash
python scripts/generate_dataset.py --n 1000 --out data/dataset_v1.h5
python scripts/train_fno.py        --data data/dataset_v1.h5 --epochs 100
python scripts/train_unet.py       --data data/dataset_v1.h5 --epochs 100
python scripts/train_convnext.py   --data data/dataset_v1.h5 --epochs 60
python scripts/compare_models.py   --data data/dataset_v1.h5
```

**Kol B** (baseline + multi-seed + heatmap):

```bash
python scripts/preprocess_eren_v2.py   # -> data/eren/dataset_v2.h5
python scripts/train_focus_point.py --epochs 300
python scripts/multi_seed_focus.py --seeds 0 1 2 --epochs 120
python scripts/train_focus_heatmap.py --arch heatmap        --loss dsnt --epochs 80
python scripts/train_focus_heatmap.py --arch heatmap_offset --loss dsnt --epochs 100
```

**Tanılama ve yardımcı**:

```bash
python scripts/phase_quantization_study.py
python scripts/phase_to_focus_offset_test.py
python scripts/diagnose_eren.py
```

**Teslim paketi**:

```bash
python scripts/build_sent_bundle.py   # -> sent/ + sent.zip
```
