# Pazartesi Toplantısı Gündemi — 28.04.2026

> Sempozyum abstract teslim deadline'ı **01.05.2026**. Bu toplantı
> deadline öncesi son danışman görüşmesi. Hedef: Kol A için final
> abstract, Kol B için yön kararı.

---

## Toplantıya kadar hazır olanlar (depodaki güncel hâl)

| İçerik | Konum |
|---|---|
| Sempozyum abstract'ları (Kol A + Kol B, EN + TR) | [`reports/abstract_*.pdf`](../reports/) |
| Klasik gold-standard kıyaslama tablosu (3 seed × 5 yöntem) | [`outputs/focus_arch_compare/gold_standard.md`](../outputs/focus_arch_compare/gold_standard.md) |
| Multi-seed mimari ablasyonu (3 omurga × 3 seed) | [`outputs/focus_arch_compare/multi_seed_summary.md`](../outputs/focus_arch_compare/multi_seed_summary.md) |
| Fizik brifingi (AI minimal anlatım) | [`reports/physics_first_brief.md`](physics_first_brief.md) |
| Model girdi/çıktı tam spec | [`reports/inputs_and_normalization.md`](inputs_and_normalization.md) |
| IEEE makale notları + niyet sorusu | [`reports/literature_notes.md`](literature_notes.md) |

---

## Tartışılacak konular (hocadan gelen sırayla)

### 1. AI'ın gerekliliği: somut sayısal gerekçe

Hocanın yorumu:
> "AI kullanmak istememizin en büyük sebebi on-the-go veri üretebilmek
> ve bunun için çok güçlü bilgisayarlar kullanmamıza gerek kalmaması."

**Bizim hazırladığımız sayısal cevap:**
- k-Wave 2-B forward (RTX 4070): **~60 s / örnek**
- FNO2d inference (RTX 4070, batch=1): **~8 ms / örnek**
- **~7 500× hızlanma**, eşit doğrulukta (test rel-L2 = 0.097)
- Inference için 100 W'lık tüketici-sınıfı kart yeterli; klinikte
  iş istasyonu kurulumuna gerek yok.

**Tartışma noktası:** abstract'ta bu rakamların öne çıkmasını
istiyor musunuz? Şu an abstract_a_en.pdf'in motivasyon bölümünde,
ama ön plana daha açık taşıyabiliriz.

### 2. Kol B — gold-standard kıyaslama (yeni!)

Hocanın yorumu:
> "Sizin gold standardınız nedir? ... literatürde kullanılan gold
> standard yöntem sonuçları ile karşılaştırılması bir zorunluluk."

**Pazartesi öncesi koşturduğumuz kıyaslama:** 5 closed-form klasik
yöntem (argmax, weighted centroid, threshold-centroid, parabolic
refinement, Gaussian-smooth+argmax) × 3 seed × aynı 22/4/4 split.

| Yöntem                     | Lateral X (mm) | Lateral Y (mm) | Wallclock |
|----------------------------|---------------:|---------------:|-----------|
| argmax(Q)                  | 39.93          | 35.26          | 0.5 ms    |
| **weighted centroid**      | **13.82**      | **14.23**      | 49 ms     |
| threshold (>0.85·Qmax)     | 37.62          | 33.98          | 53 ms     |
| parabolic refinement       | 39.95          | 35.11          | 0.6 ms    |
| Gaussian-smooth + argmax   | 37.63          | 37.49          | 37 ms     |
| **FocusPointNet**          | **4.63**       | **2.69**       | 9 ms      |

**Net bulgular:**
- Tüm peak-finder yöntemler ısı tepesini buluyor; tepe kırılma
  yüzünden hedeften 30-50 mm öteleniyor → ~70 mm RMS.
- Sadece weighted-centroid bias'ı ortalayarak makul (~33 mm).
- FocusPointNet **lateral X 3.0×, Y 5.3×** kazanım, ayrıca 5×
  daha hızlı.
- Bu, hocanın "ground truth yok" yorumuna karşı **AI'ın sistemik
  bias'ı öğrenmesi** olarak yorumlanabilir.

**Tartışma noktası:** bu kıyaslama hocanın "literatür gold standard"
beklentisini karşılıyor mu, yoksa tam k-Wave forward üzerinde
iteratif faz optimizasyonu (CG/adjoint) ile de kıyas mı bekliyor?
İkincisi 1 hafta+ ek iş, sempozyum deadline'ı için riskli.

### 3. Veri seti seçim gerekçesi (OpenBreastUS)

Hocanın yorumu:
> "Seçilen datasetin, bu çalışma için neden uygun olduğunu da
> belirtmek güzel olur."

**Hazır gerekçe (Kol A abstract'ında):** OpenBreastUS gerçek meme
MR'larından türetilmiş; sentetik fantomların yeniden üretemediği
*in vivo* yağ / fibroglandüler empedans kontrastını korur. 8 000
fantom hacmi 1 000-örneklik mevcut çalışmamıza ölçekleme runway'i
sağlıyor (validation kaybı 200 → 1 000'de 0.81 → 0.39'a düşüyor).
Orijinal yayın ultrason CT için; HIFU planlamada **ilk kullanım**.

### 4. Veri normalizasyonu / model girdileri

Hocanın yorumu:
> "Veri normalizasyonu yapıyor musun? Model girdilerinizin tam olarak
> ne olduğunu bile bilmiyorum şu an."

**Hazır cevap:** [`inputs_and_normalization.md`](inputs_and_normalization.md)
tam spec. Özetle:
- **Kol A:** girdi `(c, ρ, α)` per-channel z-score, hedef
  `log1p(p_max)` global z-score, sonra `expm1` ile Pa'ya geri.
- **Kol B:** girdi `(log1p(Q), mask)`, hedef `(x, y, z)` metre
  per-axis z-score; analitik DAS sonrası 5° kuantizasyon + gauge fix
  `φ₀ = 0`.
- İstatistikler **yalnız train split** üzerinden (val/test sızıntı yok).

### 5. IEEE referansları — niyet doğrulaması (hocadan)

Hocanın 25.04 sabahı paylaştığı iki link
([5611687](https://ieeexplore.ieee.org/abstract/document/5611687) +
[6256735](https://ieeexplore.ieee.org/abstract/document/6256735))
**alıcı-taraf B-mode adaptif beamforming** (EIBMV/ESMV) makaleleri,
HIFU değil. Üç olasılık var (detay:
[`literature_notes.md`](literature_notes.md)):

1. Yanlış doc-id paylaşılmış olabilir (yakın eşleşenler: 8370695
   MRI-uyumlu HIFU faz dizisi, 8882377 elektronik HIFU faz dizisi).
2. Adaptif beamforming heritage'ını referans olarak vermek istiyor.
3. Receive vs transmit tradeoff'unu paralel kurmamızı bekliyor.

**Toplantıda doğrudan sor.**

### 6. Kol A novelty — TUSNet vs bizim çalışma

Hocanın yorumu:
> "A kolu daha straightforward ama daha önce yapılmamış değil
> anladığım kadarıyla, orada da literatürden farkımızın iyi
> anlatılması gerekiyor."

**Bizim konumumuz (abstract'ta):** önceki ultrason sinir-operatörü
çalışmaları (TUSNet 2022, DeepTFUS 2023, Stanziola 2023) meme dışı
doku ya da sentetik fantom kullandı. Bizim yenilik:
- **Heterojen meme dokusu spesifik** OpenBreastUS üzerinde benchmark
- **Üç omurgalı ablasyon** (FNO vs U-Net vs ConvNeXt) — spektral
  önceliğin baskın etki olduğunun ampirik kanıtı, jenerik CNN'lerin
  bu görevde başarısız olduğu (ConvNeXt 0.99 LpLoss)

Bu net mi, yoksa daha keskinleştirilmesi gereken bir nokta var mı?

### 7. ABD'deki hoca ile co-author meselesi

Hocanın 24.04 mesajı:
> "abstractları hazırladıktan sonra Amerika'daki hocaya gönderip
> görüşlerini alıp bir parçası olmak ister mi diye sormayı
> planlıyorum."

**Bizim için fark etmez** — itiraz yok. Hocanın takdirine bırakıyoruz.
Sadece zaman riski var (hocadan cevap 1 Mayıs öncesi gelmeyebilir).

---

## Bilinen sınırlar (toplantıda dürüstçe konuşulacak)

[Tam liste: README.md "Bilinen Sınırlar"](../README.md#bilinen-sınırlar-limitations)

Kısa özet:
1. Eksenel (Z) doğruluğu klinik için yetersiz (25-26 mm) — veri
   sınırı, mimari değil.
2. Kol A henüz doğrudan klinik karar üretmiyor — tam pipeline
   entegrasyonu sonraki iterasyon.
3. k-Wave referansı 2-B düzlem dilim varsayımıyla.
4. Gauge fix konvansiyonu (`φ₀ = 0`) keyfi — donanım entegrasyonunda
   sıfır referansıyla hizalanmalı.
5. Klasik gold-standard kıyaslamamız closed-form ile sınırlı.

---

## Pazartesi sonrası — sonraki adımlar

Toplantıda alınacak kararlara göre değişebilir, baz öneri:

- [ ] Hocanın geri bildirimleri ile abstract'ları finalize et
  (deadline 30 Nisan'a kadar)
- [ ] ABD'deki hocaya iletim için temiz versiyon hazırla
- [ ] Hocanın istediği ek değişiklik varsa README/abstract revize
- [ ] **Sempozyum sonrası** (paper aşaması): SAM-Med3D / MONAI
  transfer learning denemesi + Eren'in 500+ örneği gelince
  multi-seed re-run

---

*Hazırlayan tarih: 25.04.2026.
Bu doküman güncel kalmalı; toplantıdan sonra notlarla güncellenecek.*
