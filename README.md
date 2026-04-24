# Soft-Tissue — Heterojen Meme Dokusunda HIFU Planlaması için AI Pipeline'ı

Yüksek-yoğunluklu odaklanmış ultrason (HIFU) ile **tümör ablasyon
planlamasının** iki pahalı hesaplama adımını (ileri basınç alanı
tahmini + ters faz sentezi) sinir operatörü + kompakt 3-B CNN ile
hızlandıran iki-kollu pipeline.

> Sempozyum makalesi için hazırlanan çalışma ortak deposu. Ana
> işbirlikçi: Eren (ITÜ) — ters problem simülatörü ve k-Wave
> doğrulaması. Danışman: Danışman.

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
| Teslim paketi | ✅ `sent.zip` (5.2 MB, 21 dosya) |
| Sonraki iterasyon: transfer learning + 500 örnek | ⏳ Plan netleşti — [future_work_ai.md](reports/future_work_ai.md) |

---

## Ne Yaptık?

İki koldan ilerledik (forward fizik hızlandırma + inverse faz planlama).
Adım adım:

1. **İlk denediğimiz yaklaşım (256 faz doğrudan tahmini) öğrenmedi.**
   Modele "istediğim hedef bu, bana 256 elemanın faz açılarını ver"
   dediğimde loss düşmeyi bırakıyordu. Biraz kurcalayınca iki yapısal
   sebep ortaya çıktı:
   - **Aynı hedef için çok farklı faz setleri aynı işi yapıyor** —
     verideki en yakın komşu hedeflere ait faz vektörlerinin benzerliği
     **0.002 ± 0.043** (gürültüden ayırt edilemez). Bu tek-çoğa bir
     eşleme; tek bir deterministik ağ bunu fit edemez.
   - **Tüm fazlara sabit bir açı (meselâ +20°) eklenince odak
     kımıldamıyor** — sistemin sürekli bir faz-ofset serbestliği var
     (sinyal işleme literatüründe bu bir *gauge* / global faz
     belirsizliği). Ağın sonsuz eşdeğer çıktıdan birini "seçmesi"
     gerekiyor ki kararsız.

   Bu iki patoloji eğitim eğrisine bakarak görülmüyor (loss biraz
   düşüyor sonra plato = "belki modeli biraz büyütsem" izlenimi). Ancak
   tanılama metrikleri yazdıktan sonra net oldu.

2. **Formülasyonu değiştirdik — pipeline'ı ikiye böldük.** Fiziğin
   sağlam olduğu yerde fiziği koruduk, AI'ı yalnızca somut kazanç
   verdiği yerde devreye aldık:
   - **Kol A — 2-B ileri basınç alanı vekili.** Doku haritasından
     (ses hızı / yoğunluk / soğurma) k-Wave basınç alanını FNO ile
     tahmin ediyoruz. FNO'nun spektral (FFT temelli) yapısı dalga
     denkleminin Green fonksiyonuna doğrudan uyuyor.
   - **Kol B — 3-B ters odak-nokta regresyonu.** Modelden 256 faz
     yerine **odak koordinatını (x, y, z)** istiyoruz. 3 serbestlik
     derecesi → tek-çoğa ve gauge sorunları ortadan kalkıyor.
     Transdüser fazları bu noktadan **analitik delay-and-sum
     hüzmelendirme** ile üretiliyor (bilinen kapalı-form).

3. **Mimari karşılaştırmalarını yaptık.** Kol A'da üç model (FNO /
   U-Net / ConvNeXt), Kol B'de beş model (düz CNN / ResNet-3D /
   çok-ölçekli UNet-encoder / ısı-haritası DSNT / ısı-haritası+offset).
   Seed gürültüsünü elemek için **Kol B'yi 3 farklı seed × 3 omurga =
   9 bağımsız koşu** ile tekrarladık (120 epoch/koşu). Ayrıca 2025
   sempozyumlarında çıkan **nnLandmark / H3DE-Net** tarzı ısı-haritası
   regresyonunu da ablasyona dahil ettik (detaylar aşağıda).

4. **Gauge serbestliğini üç bağımsız yoldan doğruladık** — sinyal
   işleme tarafında temel ama deneyle pekiştirmek gerekiyor:
   - **Analitik** — delay-and-sum denkleminden türettim (`+δ` fazdan
     faktör olarak çıkıyor, girişim desenini değiştirmiyor).
   - **Sentetik** — 256 elemanlı düzlem dizide en-küçük-kareler
     faz-plus-offset fit'i: `+20°` için yoğunluk değişimi **%1'in
     altı**, odak kayması **0 mm**.
   - **Tam-dalga k-Wave** — Eren aynı testi tam fizik simülatöründe
     tekrar etti: yine **0 mm kayma, %1'in altı yoğunluk**.

   Bu mutabakat pratik sonuç veriyor: gelecekteki her faz regresörünün
   çıktı uzayını gauge'la bölmek gerekiyor (meselâ `phase[0] = 0`
   sabitleyerek). Bu sahte bir serbestlik derecesini atıp optimizasyonu
   temizliyor.

5. **Faz kuantizasyonunu karakterize ettik** — gerçek transdüser
   sürücüleri fazı ayrık adımlarla yuvarladığı için bu hassasiyet
   kaybını ölçtük: **5° adımda akustik yoğunluk hatası %0.35**,
   odak kayması **0.1 mm'nin altı**. Bu sınırlar içinde kaldığımız
   için **5°'yi pipeline varsayılanı** yaptık. Gauge fix + 5°
   kuantizasyon birlikte, hem öğrenmeye uygun hem fiziksel olarak
   sadık bir **ayrık, gauge-sabit çıktı uzayı** veriyor.

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
- **Ana görsel rapor**: [`reports/sonuclar.pdf`](reports/sonuclar.pdf).
- Teslim edilecek paket: `sent.zip` (kök dizinde,
  `scripts/build_sent_bundle.py` tarafından yeniden üretilebilir).

## Danışmanın Son Taleplerine Cevap

Grup yazışmasındaki son mesajlarda dört somut talep vardı; durum
maddeler halinde:

### 1. Kapsam — sempozyumda 2D önceliğinin korunması

> **Danışman [10:23, 12.04.2026]:**
> "Sempozyum açısından 2D yeterdi diye düşünüyorum, böyle konuşmamış mıydık"

Haklısınız, ana mesajı 2D üzerine kurduk. **Kol A (2-B FNO)
kendi başına tam sonuç veren, figürleri makaleye gidecek nitelikte bir
çalışma** (test rel-L2 **0.097**, U-Net'in 2.7 katı daha iyi). Kol B
(3-B odak-nokta) abstract'ta **ikincil / tamamlayıcı katkı** olarak
duruyor, çünkü Eren'in 30 örneklik veri setiyle lateral 5 mm elde
ettik ama eksenel tarafta 500+ simülasyonu bekliyor. Kısacası paper'ın
omurgası 2D, 3D bonus.

### 2. Introduction + literatür taraması + novelty

> **Danışman [10:24, 12.04.2026]:**
> "Bence introduction kısmına şimdiden başlayın"
> "Sizin yaptığınız işe benzer literatür varsa şimdiden öğrensek ve
> noveltymizi ona göre kurgulasak daha iyi"

Literatür taraması yapıldı. İki karşılaştırma ekseni net:

- **İleri simülasyon hızlandırma (Kol A bizim konumumuz):** TUSNet
  (Shenoy ve ark. 2022), DeepTFUS (Yıldız ve ark. 2023), Stanziola ve
  ark. 2023. Hepsi 2-B meme dışı doku, U-Net veya FNO varyantı. Bizim
  farkımız OpenBreastUS + 1000 örnek + üç omurgalı ablasyon.
- **HIFU için öğrenilmiş ters problem (Kol B bizim konumumuz):**
  Literatür görece seyrek. Klasik optimizasyon (Mendez Peñalver
  2022) + az sayıda learning-based çalışma. Bizim farkımız 3-DOF
  yeniden formülasyon + analitik delay-and-sum hibrit.

**Novelty üç maddede kristalleşti** (abstract'ta da bu sıra):

1. **OpenBreastUS'un HIFU planlamada ilk kullanımı** (orijinal yayın
   ultrason computed tomography için).
2. **Gauge serbestliği + faz kuantizasyonu ampirik karakterizasyonu** —
   hem bizim regresörümüz hem klasik optimizatörler için bağımsız
   değer taşıyor.
3. **Örnek-verimli odak-nokta ağı + analitik hüzmelendirme hibrit
   pipeline'ı** — gerçek zamanlı, hastaya özel planlama için yapı
   taşı.

Introduction için hazır materyal:
[`reports/abstract_en.md`](reports/abstract_en.md) +
[`reports/sonuclar.pdf`](reports/sonuclar.pdf).

### 3. Haftasonu modeline ve sonuçlara bakma

> **Danışman [21:41, 17.04.2026]:**
> "Kadir, modele sonuçlara vs bir ara bakmak istiyorum. Haftasonu
> müsait bir zamanın var mı?"

Müsaitim, Cumartesi ve Pazar uygun. Toplantıya hazır malzemeler:

- **Bu depo** — GitHub'dan doğrudan dosya dosya gezebilirsiniz
  (figürler `outputs/` altında, scriptler `scripts/` altında).
- **`sent.zip`** (5.2 MB, 21 dosya) — tüm figürler + abstract + multi-seed
  tabloları + visual rapor, tek paket.
- **[`reports/sonuclar.pdf`](reports/sonuclar.pdf)** — ana
  görsel rapor, ekrana paylaşmaya uygun.
- Canlı demo için eğitilmiş modeller yerel makinede hazır; istenirse
  yeni bir girdide inference çalıştırabilirim.

### 4. Sempozyum AI makalelerinde "neyi ne ölçüde tartışmışlar"

> **Danışman [21:44, 17.04.2026]:**
> "Kadir, bu sempozyumda yayınlanan yapay zeka maklelerine bakmanı
> istiyorum. Neyi ne ölçüde tartışmışlar bilelim."

İki paralel incelememi yaptım:

- **Mimari / yöntem tarafı** — hangi modeller gerçekten işimize yarar,
  hangileri moda ama uygun değil (YOLO örneği), cutting-edge rakiplerin
  (Transolver, GNOT, SwinUNETR, U-Mamba, SAM-Med3D) bizim problemimize
  ne getireceği, ve **en büyük kazancın mimariden değil transfer
  learning'den** gelmesinin sayısal gerekçesi:
  ➜ [`reports/future_work_ai.md`](reports/future_work_ai.md)
- **Sempozyumun önceki yıllarında yayımlanan AI makalelerinin
  incelemesi** — neyi ne ölçüde tartıştıkları, bizim üç-maddelik
  novelty listemizin hangi boşluğu kapattığı tarafı **önümüzdeki hafta
  tamamlanacak**. Bu, introduction taslağıyla eş zamanlı yürütülecek
  kısım; şu an Kol B'nin ampirik tablosu oturmadan kurgu yapmamak
  için bilinçli olarak sona bıraktım.

Geri bildirim ve düzeltme olursa toplantıda not alırım.

---

## Teknik Detaylar

Bu README proje özetidir. Daha ayrıntılı teknik anlatım (ilk yaklaşımın
neden başarısız olduğu, gauge simetrisi türetimi, ısı-haritası
ablasyonunda üç ampirik bulgu, mimari seçim gerekçeleri, tam tablolar
ve reproduksiyon adımları) ayrı bir dosyada:

➜ **[`reports/teknik_detaylar.md`](reports/teknik_detaylar.md)**

İlgili dosyalar:

- [`reports/abstract_en.md`](reports/abstract_en.md) — sempozyum abstract (İngilizce)
- [`reports/abstract_tr.md`](reports/abstract_tr.md) — sempozyum abstract (Türkçe)
- [`reports/sonuclar.pdf`](reports/sonuclar.pdf) — ana görsel rapor
- [`reports/future_work_ai.md`](reports/future_work_ai.md) — cutting-edge mimari yol haritası
- [`outputs/focus_arch_compare/multi_seed_summary.md`](outputs/focus_arch_compare/multi_seed_summary.md) — multi-seed ablasyon ham tablo
- [`outputs/focus_arch_compare/summary.md`](outputs/focus_arch_compare/summary.md) — Kol B konsolide özet
