# Fizik Brifingi — HIFU Planlamasında Forward + Inverse Problem ve AI'ın Yeri

Bu doküman projenin **fizik temellerini AI bahsetmeden** anlatır.
Hedef: hocanın "yapılan işin fiziğini tam olarak anlamam gerekiyor"
yorumuna karşılık, fizikten başlayıp AI'ın **hangi noktada** ve
**neden** devreye girdiğini açıkça göstermek. Sayısal motivasyon ve
sınırlamalar somut sayılarla.

---

## 1. Klinik bağlam

Yüksek yoğunluklu odaklanmış ultrason (HIFU), tümörü çevre dokuya zarar
vermeden termal olarak yok eder. Anahtar problem: **fokal noktayı
hastaya özel doku heterojenliği altında milimetrik doğrulukla
yerleştirmek.** Yanlış konumlanma sağlıklı dokuyu yakar veya tümörü
ıskalar. Mevcut klinik iş akışı tedavi öncesi simülasyona dayanır;
bu simülasyon yeterince hızlı koşmazsa interaktif planlama mümkün
değildir.

## 2. Forward problem — akustik basınç alanı

### 2.1 Yöneten denklem

Heterojen, soğurucu ortamda lineer akustik dalga denklemi:

```
( ∂²/∂t² − c(x)² ∇² + α(x) ∂/∂t )  p(x, t)  =  S(x, t)
```

- `c(x)`  : doku-spesifik ses hızı (sıvı ~1480, yağ ~1450, fibroglandüler
  ~1510 m/s, kemik ~3000 m/s)
- `α(x)`  : frekansa-bağımlı soğurma katsayısı
- `ρ(x)` : yoğunluk (basınç ↔ partikül hızı bağlantısı)
- `S(x, t)` : transdüser kaynak terimi

Ayrıştırılmış zaman-domeninde (k-Wave gibi pseudo-spektral çözücülerin
yaptığı şey) çözüm CFL koşulu altında ilerletilir. CFL:

```
Δt  ≤  Δx / (sqrt(D) · c_max)         (D = 2 veya 3 boyut)
```

Bu bağıntı, **çözünürlüğü arttırdıkça hem grid noktası hem zaman
adımı katlanmasıyla cezalandırılan O(N⁴)** maliyet doğurur (3-D'de).

### 2.2 Hesaplama maliyeti — ölçülmüş

Bizim setup'ımızda (RTX 4070, k-Wave-Python GPU backend):

| Konfigürasyon                             | Wallclock per örnek |
|-------------------------------------------|--------------------:|
| 2-B 316×256 grid, 1 MHz, 60 µs sim          | **~60 saniye**      |
| 3-B 192×192×192 grid, 1 MHz, aynı süre      | ~25-40 dakika       |

Klinik anlamda interaktif planlama **<1 saniye yanıt** ister
(odaklamayı oynatırken anlık geri bildirim). 60 saniyelik bir
forward simülasyonu klasik anlamda planlama döngüsünde kullanılamaz.

### 2.3 Klasik hızlandırma yöntemleri ve sınırları

- **Frekans-domeni Helmholtz çözücü** — heterojen iter. çözüm 5-15 s,
  tek frekans varsayımı, transient bilgisi kaybolur.
- **Ray-tracing yaklaşımı** — milisaniyeler, fakat kırılma kuyruğunu
  kaybeder; HIFU'da meme dokusunda kabul edilemez.
- **Önceden hesaplanmış lookup tablosu** — odaklamanın konumu /
  derinliği değişince geçersiz.

Hiçbiri **on-the-go, GPU bağımsız, hastaya-özel** kombinasyonunu
sağlamıyor. Bu boşluk **forward AI vekil modelinin gerekçesidir** —
AI bir hız hilesi değil, klasik yöntemlerin **doğruluk-hız Pareto
sınırını** geçemediği bir yere yerleşiyor.

## 3. Inverse problem — faz vektörü sentezi

### 3.1 Hedef

Klinisyen istediği fokal nokta `r₀ = (x, y, z)`'yi seçer.
256 elemanlı bir transdüser dizisinin her elemanı için faz açısı
`φᵢ ∈ [0, 2π)` istenir, öyle ki dizinin ürettiği basınç alanının
maksimumu `r₀`'da olsun.

### 3.2 Klasik gold-standard yöntemler

| Yöntem                                       | Maliyet (1 hedef)            | Doğruluk |
|----------------------------------------------|------------------------------|----------|
| **Geometrik delay-and-sum** (homojen ortam)  | <1 ms (kapalı form)         | Heterojenlikte sapma 5-15 mm |
| **Pseudoinverse on Green's function**        | Saniyeler (matris ters)      | Tam dalga doğru, ortam-spesifik |
| **Iterative CG on full forward solver**      | Dakikalar / saatler          | En iyi doğruluk, klinik için yavaş |
| **Time-reversal beamforming** (deneysel)     | k-Wave forward süresi        | Heterojenlikte iyi, alan ölçümü gerekir |

Bizim "gold-standard" karşılaştırma için planlanan referans:
**iteratif faz optimizasyonu üzerinde k-Wave forward** (test split, aynı
30 örnek, wallclock + lateral RMS) — bu ölçülmemiş hâliyle Kol B
abstract'ı için eksik bir karşılaştırma maddesidir.

### 3.3 Doğrudan AI-tabanlı faz regresyonu neden çöktü?

Naif yaklaşım: ağa `r₀ → (φ₁, …, φ₂₅₆)` öğret. Bu yaklaşım iki yapısal
engelle çarpıştı.

**Engel 1 — Tek-çoğa eşleme.**
Aynı `r₀`'yı üreten **çok sayıda farklı faz seti vardır**. Verideki
en yakın komşu hedeflere ait faz vektörlerinin dairesel-kosinüs
benzerliği:

```
sim_cos = 0.002 ± 0.043     (gürültü düzeyinde)
```

Tek bir deterministik regresörün bu çoklu çözüm uzayını fit etmesi
mümkün değil — hangi temsilciyi seçeceği belirsiz.

**Engel 2 — Sürekli gauge serbestliği.**
Tüm fazlara aynı sabit ekleyince:

```
φᵢ → φᵢ + δ        ⇒        S(x, t) → e^{jδ} S(x, t)
```

Lineer akustik denklem, kaynaktaki bu global faz çarpanını çözüm
genliğinin global fazına aktarır; **basınç şiddet alanı** (dolayısıyla
odak konumu ve ısı birikimi) **kesin olarak değişmez**. Doğrulama:

| Doğrulama yolu                          | Odak kayması | Yoğunluk değişimi |
|-----------------------------------------|--------------|-------------------|
| Analitik (delay-and-sum türetimi)         | 0 mm         | %0 (tam)          |
| Sentetik 256-eleman düzlem dizi (LS fit)  | 0 mm         | <%1               |
| **Tam-dalga k-Wave** (ITÜ ortağımız)    | **0 mm**     | **<%1**           |

Yani sistemin **gerçek bir sürekli simetrisi var**. Sabit-hedef kaybı
bu durumda kararsız: ağ sonsuz eşdeğer çıktıdan birini "seçmek"
zorunda, hangi seçim en az kayba yol açar belirsiz.

### 3.4 Pivot — neden 3-DOF odak regresyonu?

Bu iki engelin ortak çözümü: çıktıyı **gauge orbit'i altında değişmez
bir niceliğe çevirmek**. Odak konumu `r₀ = (x, y, z)` tam da bu
özelliğe sahip:

- `r₀` sürekli fiziksel konumdur (mm cinsinden); ground truth tek
  değerli ve gözlemlenebilir.
- `r₀ → (φ₁, …, φ₂₅₆)` haritası tek-çoğa olabilir, ama
  `(c, ρ, α, Q) → r₀` haritası **fonksiyondur**. AI bu fonksiyonu
  öğrenir.
- Fazlar `r₀`'dan analitik **delay-and-sum hüzmelendirme** ile
  üretilir (kapalı form, <1 ms). Heterojenlik düzeltmesi gerekirse
  bir adım iter. CG eklenir; bu da gauge'tan bağımsız çalışır.

Yani Kol B'nin gerçek sorduğu şey: "verilmek istenen ısı dağılımı
göz önüne alındığında, **fokus nereye düşmeli**?" — fizik anlamında
iyi tanımlı, single-valued bir soru.

## 4. Faz kuantizasyon — donanım gerçeği

Gerçek faz-kontrol elektroniği fazları **ayrık adımlarla** sürer:
genelde 5° / 10° / 15° (FPGA / DAC çözünürlük tradeoff'una göre).
Bu kuantizasyon klasik gold-standard veya AI-tabanlı her iki
yaklaşım için de geçerli bir kısıttır.

Ölçtüğümüz hassasiyet kaybı (256-eleman dizide, 30 hedef üzerinden):

| Adım büyüklüğü | Akustik yoğunluk hatası | Odak kayması |
|----------------|-------------------------|--------------|
| 5°             | <%0.35                  | <0.1 mm      |
| 10°            | <%1.1                   | <0.3 mm      |
| 15°            | <%2.5                   | <0.7 mm      |

5° adımda hata pratik olarak kaybolduğu için **5° proje varsayılanı**.
Bu, hem öğrenme uzayını sonludaştırır (gelecekteki faz regresörleri
için 256×72 boyutlu ayrık çıktı), hem de gauge-fix (ör. `φ₁ = 0`
sabitlemesi) ile birleşince **iyi-koşullu, fiziksel olarak sadık**
bir çıktı uzayı verir.

## 5. AI'ın işlevselliği — sade çerçeve

| Alt-problem        | Klasik yöntem                                | AI yaklaşım            | Hız kazancı (beklenen) |
|--------------------|----------------------------------------------|------------------------|------------------------|
| Forward (Kol A)    | k-Wave full-wave (60 s)                      | FNO inference (~8 ms)  | **~7500×**             |
| Inverse (Kol B)    | İter. faz optimizasyonu (saniyeler-dakikalar) | FocusPointNet + DAS (<10 ms) | **>100×** (test edilecek) |

**AI burada bir akıl değil, bir tablo aramasıdır.** Ağ büyük bir tablo
gibi davranır: önceden gerçekleştirilmiş binlerce simülasyondan
öğrendiği `(girdi → çıktı)` eşlemesini interpolasyonla genişletir.
Klasik tablodan farkı, kontinyu girdiyi (her doku haritası benzersiz)
yine kontinyu çıktıyla cevaplayabilmesi — yani **online lookup**.

Bu çerçeveden bakınca AI seçimi klinik gereklilikten çıkıyor:
- **on-the-go**: hastayı taradıktan saniyeler sonra plan
- **GPU bağımsızlığı**: inference saatte 100W'lık bir kart yeterli
- **wallclock**: klasik forward'ı iki büyüklük mertebesi geçmek

Hocanın söylediği gibi: bu makalenin sunduğu şey "AI sihir yapıyor"
değil, "**simülasyon süresi-doğruluk Pareto'sundaki kör nokta için
mühendislik çözümü**".

## 6. Yarın için tartışma noktaları

Bu briefing'in pazartesi toplantısında / Eren-hoca toplantısında
açıkça konuşulması gereken yerleri:

1. **Veri seti seçimi**: Neden OpenBreastUS? (Cevap: gerçek meme
   anatomisinden türetilmiş ses-hızı haritaları, sentetik fantomlardan
   öğrenmenin generalization'a transferi vakası daha zayıf.)
2. **Veri normalizasyonu**: Track A'da `log1p(p_max)` + per-feature
   z-score; Track B'de Q için `log1p`, faz için gauge-fix `φ₁=0` +
   5° kuantizasyon. (Detaylı: `inputs_and_normalization.md`.)
3. **Gold-standard kıyaslama eksikliği**: Kol B için klasik iter. CG
   faz optimizasyonu ile **wallclock + lateral RMS** karşılaştırması
   henüz koşturulmadı. Pazar günü hazırlanacak.
4. **Kol A novelty**: TUSNet/DeepTFUS/Stanziola karşısında neden
   yeni — heterojen meme dokusu + OpenBreastUS, omurga-bağımsızlık
   kanıtı (FNO vs U-Net vs ConvNeXt).
5. **Kol B abstract'ında** doğal çerçeveleme: AI burada **forward'ı
   hızlandırma görevi yapmıyor**, **gauge'tan kurtulmuş bir
   single-valued problem**i çözüyor. Bu nüans hocaya net anlatılmalı.

---

*Bu doküman fizik perspektifinden referans niteliğindedir. Sayısal
sonuçlar ve mimari detaylar:* [`technical_details.md`](technical_details.md).
*Mimari rotası ve transfer learning argümanı:* [`future_work_ai.md`](future_work_ai.md).
