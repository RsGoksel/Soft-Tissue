# Soft-Tissue — Heterojen Meme Dokusunda HIFU Planlaması için AI Pipeline'ı

Yüksek-yoğunluklu odaklanmış ultrason (HIFU) ile **tümör ablasyon
planlamasının** iki pahalı hesaplama adımını (ileri basınç alanı
tahmini + ters faz sentezi) sinir operatörü + kompakt 3-B CNN ile
hızlandıran iki-kollu pipeline.

> Sempozyum makalesi için hazırlanan çalışma ortak deposu. Ana
> işbirlikçi: Eren (ITÜ) — ters problem simülatörü ve k-Wave
> doğrulaması. Danışman: Gülşah Hoca.

---

## Şu Anki Durum (2026-04-24)

| Kalem | Durum |
|---|---|
| 2-B ileri basınç alanı vekili (FNO) | ✅ Çalışıyor — test rel-L2 **0.097** |
| 2-B omurga ablasyonu (FNO / U-Net / ConvNeXt) | ✅ Tamam |
| 3-B ters odak-nokta regresyonu (FocusPointNet) | ✅ Çalışıyor — test RMS **26.25 mm** |
| 3-B multi-seed mimari karşılaştırması (3×3) | ✅ Tamam — üç omurga lateralde denk |
| Gauge simetrisi üçlü doğrulaması | ✅ Analitik + sentetik + k-Wave (Eren) |
| Faz-kuantizasyon çalışması (5° / 10° / 15°) | ✅ 5° pipeline'a adopte edildi |
| Isı-haritası / DSNT varyantı (nnLandmark tarzı) | ✅ Baseline ile parite (25.27 mm) |
| Sempozyum abstract (EN + TR) | ✅ Hazır — [abstract_en.md](reports/abstract_en.md), [abstract_tr.md](reports/abstract_tr.md) |
| Hocaya teslim paketi | ✅ `sent.zip` (5.2 MB, 21 dosya) |
| Sonraki iterasyon: transfer learning + 500 örnek | ⏳ Plan netleşti — [future_work_ai.md](reports/future_work_ai.md) |

---

## Ne Yaptık?

Kısaca:

1. **İlk yaklaşım (256-DOF faz regresyonu) çöktü.** Sebep yapısaldı —
   bir sonraki bölüme bakınız. Bu **negatif bulgu** pipeline'ın doğru
   yönünü belirledi.
2. **İki-kol pivotu yaptık:** Kol A ileri fiziği hızlandırır (FNO),
   Kol B ters problemi üç serbestlik dereceye (x, y, z) indirger —
   gauge simetrisi ve tek-çoğa patolojileri ortadan kalkar.
3. **Mimari karşılaştırmaları yaptık** (Kol A'da üç omurga, Kol B'de
   beş omurga) + **çok-seed'li istatistiksel ablasyon** + **ısı-haritası
   DSNT denemesi** (nnLandmark / H3DE-Net tarzı). Sonuçlar aşağıda.
4. **Gauge simetrisini üçlü yoldan doğruladık** (analitik + sentetik
   en-küçük-kareler + Eren'in k-Wave doğrulaması). `+20°` offset
   odağı kımıldatmıyor.
5. **5° faz kuantizasyon** pipeline'a adopte edildi (akustik yoğunluk
   hatası %0.35'in altı).

## Ne Bulduk? (Başarım Skorları)

### Kol A — 2-B ileri basınç alanı (1 000 OpenBreastUS örneği)

| Omurga        | Params  | Test rel-L2 |
|---------------|---------|-------------|
| **FNO2d**     | ~10.3 M | **0.097**   |
| U-Net2d       | ~7.9 M  | 0.264       |
| ConvNeXt2d    | 1.87 M  | 0.990       |

**Bulgu:** FNO'nun spektral (FFT-temelli) taban yapısı dalga
denklemine uyuyor; jenerik CNN'ler bu görevi yapmıyor. FNO U-Net'i
**2.7×**, ConvNeXt'i bir büyüklük mertebesi geçiyor.

### Kol B — 3-B ters odak-noktası regresyonu (30 hacim, 22 / 4 / 4 bölünme)

Çok-seed'li (3 seed × 3 omurga, 120 epoch/koşu) karşılaştırma:

| Omurga    | Test RMS (mm)     | X (mm)      | Y (mm)      | Z (mm)         |
|-----------|-------------------|-------------|-------------|----------------|
| **CNN**   | **26.25 ± 4.61**  | 4.63 ± 2.21 | 2.69 ± 1.21 | 25.65 ± 4.17   |
| ResNet-3D | 34.48 ± 3.74      | 5.90 ± 3.76 | 3.10 ± 1.36 | 33.64 ± 3.21   |
| UNet-enc  | 34.17 ± 8.01      | 3.93 ± 0.70 | 3.02 ± 1.05 | 33.71 ± 8.32   |
| Heatmap-DSNT | 25.27         | 4.80        | 7.04        | **23.79**      |
| Analitik (weighted-centroid) | 33.98 | 12.9 | 15.6 | 27.3            |

**Bulgular:**
- **Lateral doğruluk mimari-bağımsız** — üç CNN omurgası da 3–6 mm
  aralığında, HIFU doğal focal spot'un (~6 mm) içinde. "İyi sonuç"
  görevin özelliği, tek bir modelin değil.
- **Eksenel (Z) doğruluk mimari değil veri sorunu** — her omurgada
  25–34 mm. Z menzili lateral menzilin 2×'ü + HIFU focal zone eksenel
  olarak uzun.
- **Isı-haritası DSNT baseline'a yetişiyor** ama onu geçmiyor —
  30 örnek rejiminde sparse-supervision avantajı marjinal.

## Gelecek Çalışmalar

Kazanç potansiyeline göre sıralı:

1. **Transfer learning** (SAM-Med3D / MONAI ile pretrained 3-B
   encoder). Beklenen eksenel RMS düşüşü: 26 mm → 10–15 mm (mevcut
   veri), sonra sub-5 mm (500+ örnekle).
2. **Daha çok simülasyon** — ITÜ işbirlikçimiz 500+ örnek üretiyor.
3. **Kol A'da Transolver / GNOT** — beklenen %30–50 daha düşük rel-L2.
4. Pipeline içinde **gauge-sabit + 5° kuantizasyonlu çıktı uzayı**
   üzerinde yeniden faz regresyonu denemesi.

Ayrıntılı analiz (YOLO neden uygun değil, hangi SOTA gerçekten işe
yarar, transfer learning neden en büyük kazancı verir):
➜ [`reports/future_work_ai.md`](reports/future_work_ai.md)

## Sempozyum Abstract Durumu

- **İngilizce**: [`reports/abstract_en.md`](reports/abstract_en.md) —
  son sürüm, multi-seed + heatmap DSNT rakamlarıyla güncel.
- **Türkçe**: [`reports/abstract_tr.md`](reports/abstract_tr.md) —
  aynı içerik, Türkçe tam çeviri.
- **Ana görsel rapor**: [`reports/sonuclar_hoca.pdf`](reports/sonuclar_hoca.pdf).
- Hocaya teslim edilecek paket: `sent.zip` (kök dizinde,
  `scripts/build_sent_bundle.py` tarafından yeniden üretilebilir).

## Teknik Detaylar

Bu README proje özetidir. Daha ayrıntılı teknik anlatım (ilk yaklaşımın
neden başarısız olduğu, gauge simetrisi türetimi, ısı-haritası
ablasyonunda üç ampirik bulgu, mimari seçim gerekçeleri, tam tablolar
ve reproduksiyon adımları) ayrı bir dosyada:

➜ **[`reports/teknik_detaylar.md`](reports/teknik_detaylar.md)**

İlgili dosyalar:

- [`reports/abstract_en.md`](reports/abstract_en.md) — sempozyum abstract (İngilizce)
- [`reports/abstract_tr.md`](reports/abstract_tr.md) — sempozyum abstract (Türkçe)
- [`reports/sonuclar_hoca.pdf`](reports/sonuclar_hoca.pdf) — ana görsel rapor
- [`reports/future_work_ai.md`](reports/future_work_ai.md) — cutting-edge mimari yol haritası
- [`outputs/focus_arch_compare/multi_seed_summary.md`](outputs/focus_arch_compare/multi_seed_summary.md) — multi-seed ablasyon ham tablo
- [`outputs/focus_arch_compare/summary.md`](outputs/focus_arch_compare/summary.md) — Kol B konsolide özet
