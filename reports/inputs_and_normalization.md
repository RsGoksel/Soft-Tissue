# Model Girdileri ve Normalizasyon Spesifikasyonu

Hocanın "model girdilerinizin tam olarak ne olduğunu bile bilmiyorum"
yorumuna karşılık, **her iki kol için ağa giren ve çıkan tensörlerin
tam fizikselbirim spesifikasyonu**, normalizasyon adımları, ve
denormalizasyon yolları bu dokümanda toplandı. Yeniden üretim için
tek başına yeterli olacak şekilde yazıldı.

---

## Kol A — 2-B ileri basınç alanı vekili (FNO2d / U-Net2d / ConvNeXt2d)

### Kaynak veri seti

**OpenBreastUS** (Zeng ve ark. 2025) — 8000 anatomik meme fantomu,
gerçek meme MR/CT'den türetilmiş. Bu çalışmada bu setten **1000 örnek**
kullanıldı (rastgele, sabit seed=0).

### Girdi tensörü

```
x  ∈  ℝ^{B × 3 × 316 × 256}
```

| Kanal | Fiziksel anlam       | Birim    | Tipik aralık       |
|-------|----------------------|----------|--------------------|
| 0     | Ses hızı `c(x, y)`   | m / s   | 1450–1560          |
| 1     | Yoğunluk `ρ(x, y)`   | kg / m³ | 920–1080           |
| 2     | Soğurma `α(x, y)`    | dB/cm/MHz | 0.4–1.2          |

Spatial spacing: **0.5 mm** (her iki eksen). Toplam fiziksel boyut
**158 × 128 mm** (yaklaşık 12 cm derinlik × 13 cm lateral, su tabakası
+ doku).

### Hedef tensörü

```
y_phys  ∈  ℝ^{B × 1 × 316 × 256}
```

Tek kanal: peak akustik basınç `p_max(x, y)` (Pa). Bu, k-Wave
simülasyonunun 60 µs sürelik forward'ından zaman üzerinde alınan
maksimum mutlak değer.

### Normalizasyon adımları

**Girdi (per-channel z-score):**

```python
mean[c]  = mean over training pixels of channel c          # 3 sayı
std[c]   = std  over training pixels of channel c          # 3 sayı
x_norm   = (x - mean[c, None, None]) / std[c, None, None]
```

İstatistikler **yalnızca train split'inden** hesaplanır, val/test'e
aynısı uygulanır. Stats `dataset.PressureFieldDataset.stats` dict'inde
saklanır ve checkpoint'le birlikte kaydedilir.

**Hedef (log-uzayı + global z-score):**

```python
y_log   = log1p(y_phys)                       # log(1 + Pa) — pozitif, stabilize
mu      = mean of y_log over training pixels  # tek sayı
sigma   = std  of y_log over training pixels  # tek sayı
y_norm  = (y_log - mu) / sigma
```

`log1p` seçimi: peak basınç değerleri 1e3–1e7 Pa aralığında değişir;
log uzayında 6 büyüklük mertebesi 6 birime sıkışır, MSE/L2 kaybı
büyük örnekleri gereksiz dominate etmez.

### Denormalizasyon (görselleştirme + RMS hesaplama için)

```python
y_log_pred  = y_norm_pred * sigma + mu
y_phys_pred = expm1(y_log_pred)               # Pa cinsinden geri al
```

### Kayıp fonksiyonu

```
loss = LpLoss(d=2, p=2)(y_norm_pred, y_norm)
     + 0.3 * H1Loss(d=2)(y_norm_pred, y_norm)
```

`LpLoss` göreli L2 hatası (sample bazında normalize), `H1Loss` ek
olarak gradyan eşleşmesini cezalandırır (kırılma kuyruğu gibi yerel
yapıların korunması için).

### Test metriği

```
test_rel_L2 = LpLoss(d=2, p=2)(y_norm_pred, y_norm)
```

Raporlanan **`0.097`** (FNO) bu birimle, log-z-score uzayında. Fiziksel
basınca dönüştürülmüş örnek görseller ayrıca `outputs/fno_1k/test_sample.png`'de.

---

## Kol B — 3-B odak-noktası regresörü (FocusPointNet / Heatmap variants)

### Kaynak veri seti

ITÜ ortağımızın ürettiği **30 hacim**: HIFU forward'ı k-Wave-3D ile
koşulmuş, peak basınç alanından doku ısınma katsayısıyla türetilen
**ısı birikimi hacmi Q** ve **istenen odak noktası `r₀`** çiftleri.

Bölünme: **22 train / 4 val / 4 test** (seed=0 ile sabit permütasyon).

### Girdi tensörü

```
x  ∈  ℝ^{B × 2 × 126 × 128 × 128}
```

| Kanal | Fiziksel anlam                  | Hazırlık |
|-------|---------------------------------|----------|
| 0     | `log1p(Q)` — ısı birikimi hacmi  | `np.log1p(Q_raw)` (Q ≥ 0, Pa²·s gibi keyfi şiddet birimi) |
| 1     | Geçerlilik maskesi (binary)      | `Q > Q_max * 1e-6` (rakam küçük olan voxel'ler maskeleniyor) |

Voxel spacing: **2 mm × 2 mm × 2 mm**. Toplam fiziksel hacim
**252 × 256 × 256 mm**.

### Hedef tensörü

```
y_phys  ∈  ℝ^{B × 3}     # (x, y, z) metre cinsinden
```

Tipik aralıklar:

| Eksen | Aralık (m)        | Anlam       |
|-------|-------------------|-------------|
| x     | −0.025 … +0.025   | lateral 1   |
| y     | −0.021 … +0.025   | lateral 2   |
| z     |  0.063 …  0.148   | derinlik    |

### Normalizasyon

**Hedef (per-axis z-score):**

```python
tgt_mean[a] = mean over training samples of y_phys[:, a]    # 3 sayı, m
tgt_std[a]  = std  over training samples of y_phys[:, a]    # 3 sayı, m
y_norm      = (y_phys - tgt_mean) / tgt_std
```

**Girdi:** `log1p` zaten lineer-olmayan sıkıştırma yaptığı için ek
z-score kullanılmadı. Kanal-0'ın istatistik aralığı log(1+...) sayesinde
~[0, 5] mertebesinde kalır.

### Kayıp + test metriği

```
loss   = MSE(y_norm_pred, y_norm)             # eğitim
err_mm = (y_phys_pred - y_phys) * 1000        # mm cinsinden
RMS    = sqrt( mean over test of (err_mm**2).sum(axis=1) )
per_axis_RMS = sqrt( mean over test of (err_mm**2)[:, a] )
```

Raporlanan **26.25 ± 4.61 mm** (multi-seed CNN baseline) bu birimle.
Lateral RMS X için 4.63 ± 2.21 mm, Y için 2.69 ± 1.21 mm.

### Faz çıktısı (analitik adım, AI değil)

Eğitim modelin son ürünü değildir — **transdüser fazları** ayrıca
hesaplanır:

```python
# 256-eleman dizinin geometrik konumu r_i bilinen sabit
# r0_pred = model çıktısı (m)
delays  = ||r_i - r0_pred||  /  c_water        # saniye
phases  = (2 * pi * f * delays)  mod  (2 * pi)
phases  = round(phases / 5°) * 5°              # 5° kuantizasyon
phases -= phases[0]                             # gauge-fix: φ_0 = 0
```

Yani **AI yalnızca odak noktasını verir**, fazlar kapalı-form
hüzmelendirme + iki ek post-processing adımı (kuantizasyon + gauge fix)
ile elde edilir.

---

## Heatmap varyantı için ek bilgi (Kol B alternatif yol)

DSNT-tabanlı heatmap variantı için iç temsil:

```
v_voxel = affine_fit(y_phys)               # (B, 3) voxel koordinat
H_target = Gaussian(v_voxel, σ=6 voxel)    # (B, 1, 126, 128, 128)
loss_dsnt = MSE(soft_argmax(H_pred), v_voxel) + 1e-4 * variance_reg
```

Affine fit = 22 train örneğinin Q'sundan hesaplanmış 0.85-eşik focal
voxel ↔ `y_phys` arasındaki en küçük kareler eşlemesi (bkz.
`focus_heatmap.fit_voxel_affine`). Train fit'in residual RMS'i 16
voxel = 32 mm; bu, heatmap-tabanlı yaklaşımın bizim 22-örnek
rejimindeki *targetability ceiling*'ini belirler.

---

## Hızlı referans

| Soru | Cevap |
|---|---|
| Kol A girdi şekli? | `(B, 3, 316, 256)`; kanallar `c, ρ, α` |
| Kol A çıktı şekli? | `(B, 1, 316, 256)`; peak basınç (Pa, log-z-score uzayı) |
| Kol B girdi şekli? | `(B, 2, 126, 128, 128)`; kanallar `log1p(Q)`, mask |
| Kol B çıktı şekli? | `(B, 3)`; odak `(x, y, z)` (m, z-score uzayı) |
| Hangi normalizasyon train split'inden? | İkisi de — sızıntıyı önlemek için |
| Voxel spacing? | A: 0.5 mm, B: 2 mm |
| Faz kuantizasyon adımı? | 5° |
| Gauge sabitleme? | `φ_0 = 0` (DAS sonrası) |

Reproduksiyon adımları: [`technical_details.md` §10](technical_details.md#10-reproduksiyon).
