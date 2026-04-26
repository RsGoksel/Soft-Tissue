# Literatür Notları — Hocanın Önerdiği IEEE Makaleleri

Hoca 25.04.2026 sabahı iki IEEE Xplore linki paylaştı. Aşağıda her ikisinin
metadatası, içerik özeti ve **bizim çalışmamızla bağlantısı** var.
Pazartesi toplantısında **niyet doğrulaması** yapılması gerekiyor —
ikisi de doğrudan HIFU faz kuantizasyonu makalesi değil; *receive-side*
diagnostik B-mode adaptif hüzmelendirme makaleleri. Hocanın muhtemelen
benzetme kurmamızı istediği veya doc-id'lerin tıklanırken karıştığı iki
ihtimal var.

---

## 1. doc 5611687

**Başlık:** *Eigenspace-Based Minimum Variance Beamforming Applied to
Medical Ultrasound Imaging*
**Yazar:** Babak Mohammadzadeh Asl, Ali Mahloojifar
**Yıl / Dergi:** 2010, IEEE Trans. Ultrasonics, Ferroelectrics, and
Frequency Control **57(11):2381–2390**
**DOI:** 10.1109/TUFFC.2010.1706
**PubMed:** [PMID 21041127](https://pubmed.ncbi.nlm.nih.gov/21041127/)

**Özet (paraphrase):** B-mode ultrasonda alıcı-taraf adaptif
hüzmelendirme için EIBMV önerisi. Standart minimum-variance ağırlık
vektörünü kovaryans matrisinin sinyal alt-uzayına projekte ederek
hem lateral çözünürlüğü hem kontrastı (yan-lobe baskısıyla)
geliştiriyor; aynı zamanda steering-vector hizalama hatalarına dayanıklı.

**Bizim çalışmamızla bağlantı (dolaylı):**

- Bu **alıcı tarafı** (receive-side) imaging; biz **gönderici tarafı**
  (transmit-side) terapi yapıyoruz. Doğrudan analog değil.
- Yine de bağ: makalenin "EIBMV küçük per-element faz hatasına
  toleranslıdır" gözlemi, bizim **5° faz kuantizasyonunun %0.35
  yoğunluk hatası** sınırına benzer bir sağlamlık tartışmasını paralel
  okur.
- Eigenspace/kovaryans perspektifi, bizim **gauge invariance** (sabit
  +δ) bulgumuza alternatif bir matematiksel pencere açar — referans
  olarak introduction'a girebilir.

**Önerilen atıf bağlamı:** Background paragrafında veya bir contrast
örneği olarak: *"Unlike receive-side adaptive beamforming
[Asl &amp; Mahloojifar 2010], our work targets transmit-side phase
synthesis under quantization."*

---

## 2. doc 6256735

**Başlık:** *Eigenspace Based Minimum Variance Beamforming Applied to
Ultrasound Imaging of Acoustically Hard Tissues*
**Yazar:** Saeed Mehdizadeh, Andreas Austeng, Tonni F. Johansen, Sverre Holm
**Yıl / Dergi:** 2012, IEEE Trans. Medical Imaging **31(10):1912–1921**
**DOI:** 10.1109/TMI.2012.2208469
**PubMed:** [PMID 22868562](https://pubmed.ncbi.nlm.nih.gov/22868562/)

**Özet (paraphrase):** EIBMV'yi (burada ESMV adıyla) ileri/geri
mekansal ortalamayla genişletip kemik gibi akustik olarak sert
yüzeylerin görüntülenmesinde kullanıyor — speküler yansıma ve
gölgenin delay-and-sum'ı bozduğu yerlerde. Kritik parametre sinyal
alt-uzayı rank'ı: düşük rank kenarları keskinleştirir + gürültüyü
bastırır + speckle'ı bozar (çözünürlük/sadakat takası).
Simülasyon + fantom + klinik veriyle gösterilmiş.

**Bizim çalışmamızla bağlantı (dolaylı, ama daha tematik):**

- Çalışma akustik empedans kontrastı yüksek ortamlarda görüntüleme
  yapıyor; biz heterojen meme dokusunda (ses hızı 1450–1560 m/s,
  yağ-fibroglandüler kontrast) **forward** simülasyon yapıyoruz.
  Heterojenlik motivasyonu paralel.
- "Rank vs fidelity tradeoff" kavramı, bizim "**adım büyüklüğü vs hata**"
  takasımızın metodolojik analoğu (5° → %0.35, 10° → %1.1, 15° →
  %2.5).

**Önerilen atıf bağlamı:** Introduction'da heterojen ortamda adaptif
ultrason yöntemlerinin discretization/rank ↔ fidelity takasını yönetmek
zorunda olduğu cümlesi; bizim 5° faz-kuantizasyon bütçemizi transmit
tarafının analoğu olarak çerçevelemek.

---

## Önemli not — pazartesi hocaya sorulacak

İkisi de **alıcı-taraf B-mode görüntüleme** makalesi, HIFU değil. Üç
olası durum:

1. **Yanlış doc-id paylaşılmış olabilir.** Yakın eşleşen olası
   alternatifler:
   - **doc 8370695** — *MRI-compatible HIFU phased array for breast
     tumor treatment* (2018) — bizim bağlamımıza çok daha uygun
   - **doc 8882377** — *Fully electronically steerable HIFU phased
     array* (2019) — yine HIFU faz kontrolüyle birebir ilgili

2. **Hoca bilinçli olarak EIBMV/ESMV literatürünü öneriyor olabilir** —
   adaptif ultrason hüzmelendirmeyi kendi araştırma alanı olarak
   tanıttığı için. Bu durumda atıf "background / methodological
   ancestry" olarak kullanılır.

3. **Kuantizasyon argümanını adaptif beamforming dünyasına yerleştir-
   memizi istiyor olabilir** — receive tarafındaki rank-tradeoff'la
   transmit tarafındaki step-size-tradeoff'u paralel kurarak.

**Aksiyon:** Pazartesi 28.04 toplantısında **doğrudan sor:** "Hocam,
gönderdiğiniz iki link adaptif imaging beamforming makaleleri,
HIFU faz kontrolüyle doğrudan ilgili değil. Niyetiniz tam olarak
neydi — yanlış mı tıklandı, yoksa adaptif beamforming heritage'ı
mı bizim çalışmamıza bağlamamızı istiyorsunuz?"

---

## Erişim notu

IEEE Xplore HTTP 418 anti-bot bloğuyla doğrudan erişimi engelledi.
Yukarıdaki metadata PubMed (PMID 21041127, 22868562) + IEEE arama
snippet'lerinden derlendi. **Camera-ready submission için verbatim
abstract gerekirse**, kurumumuzun IEEE Xplore aboneliği üzerinden
veya `metapub`/`pyPubMed` ile PMID üzerinden çekilebilir.
