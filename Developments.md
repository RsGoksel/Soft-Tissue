<!--
# LLM AGENT TALİMATLARI

Bu dosya projenin değişiklik günlüğüdür. Her tarih bloğunda:
- **1. seviye bullet** = Sade özet (dosya adı, CSS değeri YOK)
- **2. seviye bullet** = Teknik detay (dosya yolları, ne değişti)

Etiketler: [ml] [data] [simulation] [pipeline] [dokuman] [altyapi] [test] [research]
İşlem: Oluşturuldu / Değiştirildi / Silindi / Düzeltildi / Yeniden yazıldı

Arşiv: Son 5 haftadan eski → docs/archive/development_YYYY_MM.md
-->

# Ultrason Changelog

Bu belge, **Ultrason** projesinin (yumuşak meme dokusunda ultrason basınç alanı tahmini için FNO vekil modeli) günlük güncellemelerini ve teknik detaylarını takip eder.

---

## 2026-04-17

- [ml] **FocusPointNet SONUÇLARI: Test X/Y RMS ≈ 5 mm (excellent), Z RMS 27 mm (anisotropic).** Sadece 30 sample v2 data ile, 300 epoch LR=3e-4 + gradient clipping + cosine schedule.
  - **Test MSE (standardise)**: 0.386
  - **Test RMS (physical mm)**: overall **27.89 mm**; per-axis **X 5.09 mm, Y 5.28 mm, Z 26.91 mm**
  - **Model**: `FocusPointNet` 0.89M params, 4-level 3D CNN (16→32→64→128) + AdaptiveAvgPool3d + MLP(128→64→3), input (2, 126, 128, 128) = (log_Q_norm, mask).
  - **Çıktılar**: `outputs/focus_point/best.pt`, `loss_curve.png`, `test_scatter.png` (her eksen için pred vs true scatter + diagonal).
  - **Yorum**: Lateral (X, Y) tahmin 5mm altında — bu HIFU focal spot genişliğinin altında, pratik olarak mükemmel. Z (axial, dalga propagation yönü) 27mm: HIFU focal zone'un doğal axial smear'i + Z aralığının X/Y'nin 2× olması. Daha fazla sample ile Z'nin belirgin iyileşmesi beklenir. Subagent'ın "9.9 mm RMS" diagnostic hedefine ulaşılmadı ama o val_frac=0.2 (6 sample), bizde val/test=0.15/0.15 (4/4 sample) — değerlendirme setleri farklı, rakamlar doğrudan kıyaslanmaz.
  - **İlk koşu (LR=1e-3, no clip) instabildi**: val RMS 22→577 mm arası salınım yaptı, BatchNorm + 4-sample batch + büyük LR kombinasyonu. Grad clipping (max_norm=1.0) + LR 3e-4 + 300 epoch ile stabilize edildi.
  - **Eren için bu ne demek**: Eren'in test planına uyumlu. "Hedef heatmap ver, model fazları üretsin, simulatöre at" planında artık **model fazlar yerine focus noktasını veriyor, simulator beamforming ile fazları analitik üretiyor**. Fiziksel olarak daha doğru ve çok daha sample-efficient.

---

- [ml] **KRİTİK PIVOT: Q→phases yerine Q→focus_pt (3 DOF) tahminine geçildi.** ml-engineer subagent'ın diagnostic scripti (`scripts/diagnose_eren.py`) iki fiziksel hipotezi test etti ve problemin ill-posed'luğunu matematiksel olarak kanıtladı.
  - **H1 test (Q → target_pt)**: 30 sample'da 200 epoch, val MSE 0.082 (standardize edilmiş), un-standardise edilmiş **RMS 9.9 mm**. Subagent eşiği 10 mm'nin altında = GO. Q volume hedef pozisyonunu AÇIK SEÇİK kodluyor, 30 sample'la bile regressable.
  - **H2 test (target_pt → phase vector)**: CSV'deki 200 temiz sample üzerinde, her distance bin için mean cosine similarity ölçüldü. Nearest-neighbor bin (0.0014 - 0.0272 m) için **mean_sim ≈ 0.002** (std 0.043) — **essentially RANDOM**. Daha uzak bin'lerde de 0.003-0.004, tüm bin'ler uniform dağılımdan ayırt edilemez. Anlamı: aynı target_pt için farklı simulation run'ları birbirinden tamamen farklı faz vektörleri üretiyor — Q → phases bir fonksiyon değil, bir-çoğa eşleme.
  - **Karar**: 256 faz tahmin etmeyi bırakıp **3 DOF focus point tahmin ediyoruz**. Downstream'de phases'ı standard beamforming formülü ile analitik türetiriz. Bu Eren'in test planına (model'in fazları simulatöre atarak denenmesi) birebir uyumlu — zaten beamforming simulatörde var.
  - **Oluşturuldu**: `src/eren_model.py` içinde `FocusPointNet` — 4-level 3D CNN (2 → 16 → 32 → 64 → 128 channels) + GAP + MLP head (128 → 64 → 3). ~0.8M param, sample-efficient.
  - **Oluşturuldu**: `scripts/train_focus_point.py` — proper train/val/test split, standardized target, MSE loss, cosine LR, per-axis RMS reporting in physical mm, test scatter plot (pred vs true per axis).

---

- [research] **ml-engineer subagent architectural review raporu.** 5 dosyayı inceledi (model, dataset, train, preprocessor, dataset.h5), tavsiyeleri ve kod değişikliklerini uyguladı.
  - **Subagent tespitleri**: Mevcut 3D CNN + GAP + MLP mimarisinde ciddi tasarım hatası yok. Ana sorun loss fonksiyonu ve problem formülasyonu: per-transducer phase std ≈ π/√3 ≈ uniform random → problem ill-posed olabilir. 30 sample için phase prediction "almost certainly not enough" ama asıl bariyer veri değil, phases'ın Q'nun deterministic fonksiyonu olmaması.
  - **Eklediği kod**: `src/eren_model.py` içinde `GaugeInvariantPhaseLoss` (gauge symmetry'sini yakalayan complex-inner-product loss), `scripts/diagnose_eren.py` (H1+H2 diagnostic), `scripts/train_eren_inverse.py`'a `--loss gauge` (yeni default).
  - **Subagent recommendation verdikti**: "Regardless of H1/H2 result, consider switching the prediction target to focus-point + analytic steering: 3 DOF is vastly easier than 256 and is what the physics actually constrains." H2 sonucu bu öneriyi zorunlu hale getirdi.

---

- [ml] **Eren inverse pipeline iyileştirildi: trivial-zero solution eliminated.** Smoke test sonucu 30 sample üzerinde eğitim loss 0.48'de tıkanmıştı — bu tam olarak "tüm çıktıları sıfır tahmin et" trivial çözümünün MSE skoru. Geometrik analiz: sin/cos MSE'de prediction=(0,0), target=(sin_t,cos_t) için loss = sin²+cos² = 1 per pair, ortalama 0.5. Model gradient açısından risksiz en konforlu yere düşüyor. İki değişiklik ile bu kapatıldı.
  - **Değiştirildi**: `src/eren_model.py` — Final `nn.Tanh()` kaldırıldı, yerine her (sin, cos) çiftine L2 normalization ile unit-circle projection eklendi (`sc / norm.clamp_min(1e-6)`). Output artık ham logitler yerine geometrik olarak doğru açısal vektörler. Sıfır çıktı = norm≈0 → projection kararsız, gradient anlamlı.
  - **Değiştirildi**: `scripts/train_eren_inverse.py` — Default loss `sincos`→`circular` (`1 - cos(φ_pred - φ_true)`). CircularPhaseLoss'ta prediction=(0,0) için cos_diff=0, loss=1.0 (MAKSIMUM ceza). Bu trivial çözümü kapatıyor, model gerçek bir açı tahmin etmek zorunda kalıyor.

---

- [ml] **Target_pt conditioning + standardization eklendi.**
  - **Değiştirildi**: `src/eren_model.py` — `PhaseInverseNet.__init__(use_target_pt=True)` flag'i: MLP bottleneck'ine (B, 3) target koordinatını concat eder. 100 sample'lık rejimde güçlü prior — model "bu heatmap'in hedefi şu noktada" bilgisini doğrudan alır.
  - **Değiştirildi**: `src/eren_dataset.py` — `stats['tgt_mean']`, `stats['tgt_std']` training split'inden hesaplanıyor. Target_X/Y (±2.6 cm), Target_Z (5.9-15 cm) farklı ölçeklerde — standardize edilmeden MLP input'u bozulur.
  - **Değiştirildi**: `scripts/train_eren_inverse.py` — `forward(batch)` helper'ı, `use_target_pt` true ise `batch["target_pt"]` GPU'ya gönderip `model(x, target_pt=tgt)` çağrısı. False ise normal tek-arg forward.

---

- [data] **V2 dataset: 30 sample Alg=1 Noise=0 (en temiz) subset.**
  - **Oluşturuldu**: `scripts/preprocess_eren_v2.py` — MATLAB 5.0 format için scipy.io.loadmat. Her `.mat` içinde `Q_heat_cropped (253, 256, 256) float32` + `target_pt (1, 3)`. Fazlar CSV'den join ediliyor (v1'de .mat içindeydi). 2× downsample (→ 126×128×128), log1p(Q/1e3) normalization, float16 storage, LZF compression.
  - **Veri analizi (CSV tam 3000 satır)**: Algorithm 1: 2600 row, Algorithm 2: 400. NoiseType 0: 600, 1: 1600, 2: 800. İlk 30 sample (elimizdeki) **tümü Alg=1 Noise=0** — kasıtlı hatalı örnek yok. Unique target count = 3000 (hiç duplicate yok). Per-transducer phase std **1.81 rad** = uniform random (target-phase korelasyonu çok zayıf — bu inverse problemin ill-posed kısmına işaret ediyor).
  - **Çıktı**: `data/eren/dataset_v2.h5` — 30 × (126, 128, 128) Q + (2, 256) phases_sincos + (256,) phases_rad + target_pt_m + sim_id. 124 MB.

---

- [data] **gdown rate-limit'e takıldı: 100 hedef yerine 30 dosya alınabildi.** Drive klasöründe aslında **1370+ dosya var** (Eren 3000'in büyük kısmını yüklemiş). Google Drive'ın shared-folder rate limit'i (~30 file/session) gdown'ı durdurdu. Retry, `--remaining-ok` (gdown bu versiyonda yok) ve Drive MCP search — hiçbiri 30'dan ileri gitmedi.
  - **Karar**: Mevcut 30 sample ile ilerliyoruz. Disk durumu kritik (C: 5 GB boş, 465 GB dolu) ayrıca bir faktör. Eren'den 3000 tamamlanınca transfer için WeTransfer/OneDrive share benzeri alternatif istemek lazım.

---

- [dokuman] **ml-engineer subagent çağrıldı: inverse pipeline architectural review.** Background'da çalışıyor; code review + sample-efficient enhancement önerileri + problem tractability değerlendirmesi yapacak. Response geldiğinde trivial olmayan iyileştirmeler direkt uygulanacak.

---

- [data] **Eren v2 veri seti (`dataset_cropped/`) indiriliyor + yeni preprocessor yazıldı.** Eren formatı değiştirdi: artık zip değil, tekil `.mat` dosyaları; cropped + yeniden boyutlandırılmış. `gdown` ile Drive'dan indirme devam ediyor.
  - **Format değişimi**: MATLAB 7.3 (HDF5) → MATLAB 5.0. Yeni key `Q_heat_cropped` shape `(253, 256, 256)` float32, **NaN yok**, değer aralığı 1.8e-4 .. 2.1e8 (öncekinden biraz daha temiz). `target_pt` aynı. **Fazlar artık .mat dosyasında değil — sadece CSV'de** (`hifu_phase_dataset_1mm.csv` içinde Phase_1..Phase_256). Boyut yaklaşık 60 MB/sample (Eren'in "128³" söylemine rağmen gerçekte cropped edilmiş 253×256×256).
  - **Oluşturuldu**: `scripts/preprocess_eren_v2.py` — scipy.io.loadmat ile `.mat` dosyalarını açıyor, CSV'den faz açılarını sim_id ile eşleştiriyor, Q'ya log1p + downsample 2× uyguluyor, sin/cos encoding, tek HDF5'e yazıyor. `compression='lzf'`, `chunks=(1,) + ds_shape`, float16 storage. v1 preprocessor'ın aksine tek bir mat-dir + CSV yeterli (zip streaming'e gerek yok).
  - **Değiştirildi**: `src/eren_dataset.py` — `_has_mask()` helper'ı eklendi, v2 formatında `mask` alanı olmadığı için `np.ones_like(Q)` ile tam geçerli kanal kullanılıyor. Eski v1 dataset (mask'lı) ile geriye uyumlu.
  - **Durum**: gdown `dataset_cropped/` klasöründen 100 sample'ı tek tek çekiyor (Drive rate limiti yüzünden klasör tek seferde alınamıyor), background'da devam.

---

## 2026-04-09

- [ml] **Eren kolu smoke test: model öğrenemedi (rastgele tahmin seviyesinde).** 100 sample, 60 epoch (20'de durduruldu), val phase error ~90.9° — tam rastgele seviye. Somut feedback Eren'e iletilecek: daha fazla veri (en az 1000) + problem formülasyonu tartışması (target_pt ayrı input olarak verilmeli mi? iterative refinement? cycle-consistency?).
  - **Çalıştırıldı**: `train_eren_inverse.py --data data/eren/dataset.h5 --epochs 60 --batch-size 2 --base-channels 16 --loss sincos`
  - **Metrik**: Train loss 0.494, val loss 0.508 (epoch 1-20 boyunca sabit), val phase error 89.6° → 90.9°. Model output ile target arasında sıfır korelasyon.
  - **Kök neden hipotezi**: (a) 70 training sample 1.47M paramlı 3D CNN + MLP için çok az. (b) Inverse problem ill-posed — aynı Q'yu üreten çok sayıda faz kombinasyonu var, target_pt bilgisi olmadan model referans kaybediyor. (c) 3D encoder → GAP → MLP spatial bilginin büyük kısmını atıyor.
  - **Sonraki adım**: Eren 3000 sample'ı tamamlayınca tekrar dene. Paralel olarak (Q, target_pt) iki-input mimari, veya forward (phase → Q) yönünde eğitip cycle-consistency ile ters çıkarma denenebilir.

---

- [ml] **Eren inverse pipeline yazıldı.** 3D CNN encoder + MLP head, sin/cos phase output.
  - **Oluşturuldu**: `src/eren_model.py` — `PhaseInverseNet` (4-level 3D CNN, 32→64→128→256 ch, AdaptiveAvgPool3d, MLP head 512→512→512, tanh output (B, 2, 256)). `SinCosMSELoss`, `CircularPhaseLoss`, `phase_error_degrees` metric.
  - **Oluşturuldu**: `src/eren_dataset.py` — `ErenPhaseDataset` HDF5 wrapper, RAM cache, input (2, D, H, W) = (normalize edilmiş log_Q, valid mask), target (2, 256) sin/cos. `q_max` training split'inden hesaplanıyor.
  - **Oluşturuldu**: `scripts/train_eren_inverse.py` — eğitim loop, `SinCosMSELoss` veya `CircularPhaseLoss` seçeneği, test figürleri (loss curve + target vs predicted phase plot).

---

- [data] **Eren preprocessor: 9 GB zip → 618 MB HDF5.** Stream-based processing, zip'leri hiç extract etmeden belleğe.
  - **Oluşturuldu**: `scripts/preprocess_eren.py` — 5 zip dosyasını scanliyor, her `sim_id_XXXX.mat`'i bellekte okuyup h5py ile parse ediyor, Q_heat'i 2× downsample + log1p normalize + float16 storage, phases sin/cos 2-kanallı kaydediyor, mask ayrı kanal, target_pt metre cinsinden saklanıyor. CSV ile phase cross-check yapıyor.
  - **Bug fix**: İlk çalıştırmada `Q_ds[written][...] = value` HDF5'e hiç yazmıyordu (numpy'ın advanced indexing copy-semantics problemi). `Q_ds[written] = value` olarak düzeltildi.
  - **Çıktı**: `data/eren/dataset.h5` (618 MB) — `Q (100, 155, 160, 126) float16`, `mask (100, 155, 160, 126) uint8`, `phases_sincos (100, 2, 256)`, `phases_rad`, `sim_id`, `target_pt_m`. 100 sample, sim_id 1–100. Log-space Q range [0, 12.2] (4–5 dekat dinamik aralık).
  - **Süre**: 125 sn (100 sample × 1.25 s/sample).

---

- [research] **Eren'in HIFU verisi keşfedildi.** Eren (ITU'dan ortak proje) inverse HIFU planlama çalışması için fazlar-heat çiftleri verdi. Paralel kol açıldı.
  - **İndirildi**: Google Drive'dan 5 zip (9 GB) + `hifu_phase_dataset_1mm.csv` (14 MB) + `hifu_tissue_maps_050mm.mat` (4.3 MB).
  - **İncelendi**: Her heatmap `.mat` içinde `Q_heat (310, 321, 253) float32` + `phases (1, 256) float64` + `target_pt (3, 1)`. CSV'de 3000 satır ama sadece ilk 100 için heatmap var. `TargetXYZ_m, Algorithm, NoiseType` kolonları kontrol amaçlı, modele girmeyecek. NaN %3.42 sabit (simülasyon domain dışı, "bozuk" değil). Tissue maps 3D @ 0.5 mm rezolüsyon — tüm simülasyonlarda ortak tek meme geometrisi.
  - **Problem formülasyonu**: Eren forward simülasyon (phases → Q), model **inverse**: "bu bölgeyi ısıt" → hedef Q input, 256 faz output. Pratik HIFU tedavi planlama.
  - **Karar**: Şimdilik full 3D downsample (crop etmek target_pt'yi modele sızdırırdı — leakage), 2 kanal (log_Q, mask), tissue map şimdilik kullanmıyoruz (tek geometri → bilgi vermez).

---

- [ml] **FNO vs U-Net karşılaştırma: 1000 sample üzerinde FNO 2.7× daha iyi.** Makale için final comparison figürü hazırlandı.
  - **Çalıştırıldı**: `scripts/compare_models.py --data data/dataset_v1.h5 --fno outputs/fno_1k/best.pt --unet outputs/unet_1k/best.pt --out outputs/comparison_1k.png`
  - **Metrikler (sample-wise LpLoss, lower=better)**: **FNO 0.0971, U-Net 0.2638 → FNO 2.7× daha iyi** (200 sample smoke'da 1.5×'tı, data scaling avantajı belirgin).
  - **Görsel**: `outputs/comparison_1k.png` — 5-panel (GT, FNO pred, U-Net pred, FNO error, U-Net error). FNO prediction'ı ground truth'tan görsel olarak ayırt edilmiyor; U-Net prediction'ı bulanık + blok artifactları içeriyor; FNO error map'i neredeyse beyaz, U-Net error map'inde belirgin yatay bantlar var. Makale introduction/results için birebir kullanılabilir.

---

- [ml] **U-Net baseline 1000 sample üzerinde eğitildi.** FNO ile aynı split + loss arayüzü.
  - **Çalıştırıldı**: `scripts/train_unet.py --data data/dataset_v1.h5 --epochs 40 --batch-size 4 --out-dir outputs/unet_1k`
  - **Metrikler**: 7.85M param, train 2.089, **val 1.031, test 1.042** (LpLoss). Loss eğrisi yumuşak ama plato'lamıyor, daha fazla eğitim bir miktar daha iyileştirebilir ama FNO ile aradaki gap zaten belirgin.
  - **Süre**: ~35 dk CUDA.
  - **Çıktılar**: `outputs/unet_1k/best.pt`, `loss_curve.png`, `test_sample.png`.

---

- [ml] **FNO 1000 sample üzerinde eğitildi — 200 sample smoke test'e göre %51 iyileşme.**
  - **Çalıştırıldı**: `scripts/train_fno.py --data data/dataset_v1.h5 --epochs 40 --batch-size 4 --n-modes 24 --hidden-channels 64 --n-layers 4 --loss lp+h1 --out-dir outputs/fno_1k`
  - **Metrikler**: 5.16M param, train 0.906, **val 0.394, test 0.383** (LpLoss). 200-sample smoke test val 0.81 → 1000-sample val 0.39 = %51 mutlak iyileşme.
  - **Süre**: ~40 dk CUDA (cache sonrası). İlk koşuda OOM (batch=8 × n_modes=32), batch=4 + n_modes=24 + n_layers=4 ile stabilize.
  - **Çıktılar**: `outputs/fno_1k/best.pt`, `loss_curve.png`, `test_sample.png`.

---

- [altyapi] **Dataset RAM cache eklendi — eğitim ~1.7× hızlandı.** HDF5 per-sample I/O darboğazdı (GPU %26 util), cache sonrası GPU %95+ sürekli.
  - **Değiştirildi**: `src/dataset.py` — `PressureFieldDataset.__init__`'e `cache_in_ram=True` default, `_load_cache()` tüm split'i normalize edip RAM'de tutuyor. `__getitem__` cache varsa diskten okumayı atlıyor.
  - **Değiştirildi**: `scripts/train_fno.py` — loss toplamadaki bug düzeltildi (`* x.size(0)` iki kez çarpıyordu, `tr_loss /= len(train_loader)` olarak düzenlendi — önceki rakamlar 4× şişmişti).

---

- [data] **1000 sample dataset generation tamamlandı.** `breast_train_speed.mat`'tan 1000 rastgele fantom, native path + water standoff + rastgele tümör.
  - **Çalıştırıldı**: `python scripts/generate_dataset.py --mat breast_train_speed.mat --n 1000 --grid 256 --out data/dataset_v1.h5`
  - **Çıktı**: `data/dataset_v1.h5` 1.3 GB. Layout: `inputs (1000, 3, 316, 256)`, `targets (1000, 1, 316, 256)`, `focus`, `phantom_index`. Targets range: 22 kPa → 2.4 MPa (tipik).
  - **Süre**: ~4 saat CPU (başta ~22 sn/sample, fantom yükleme 50 sn overhead).

---

- [ml] **Model iyileştirme smoke test: 2.7× kayıp azalması (aynı 200 sample).** İyileştirilmiş eğitim yalnızca mimari + loss değişiklikleriyle Val LpLoss 2.39 → 0.81.
  - **Çalıştırıldı**: `scripts/train_fno.py --epochs 20 --batch-size 4 --n-modes 32 --hidden-channels 64 --n-layers 5 --loss lp+h1 --out-dir outputs/fno_smoke`
  - **Metrik**: Train LpLoss 1.88, **Val 0.81, Test 0.90**. 11.20M param (önceki 2.06M'den 5.4× büyük).
  - **Görsel**: `outputs/fno_smoke/test_sample.png` — ground truth dalga bantları net, FNO prediction yumuşak envelope yakaladı (önceki texture-spam yerine), rel L1 ~%105 denormalize nedeniyle yanıltıcı (asıl metrik LpLoss).
  - **Sonuç**: Log-space target + H1 + büyük kapasite kombinasyonu işe yarıyor. 1000 sample'da çok daha iyi olacak.

---

- [ml] **Model pipeline iyileştirmeleri.** 200 sample + 50 epoch sonucundaki underfit'i (Val 2.39) adreslemek için 4 yönlü iyileştirme.
  - **Değiştirildi**: `src/dataset.py` — `PressureFieldDataset` log-space target scaling. `log_target=True` (default), `y_offset=1e4` (10 kPa floor). Normalize: `y' = log1p(p_max/y_offset) / log_scale`. `denormalize_target()` metodu eklendi (Pa birimine geri dönüş). `_compute_stats()` log veya linear mode destekliyor. Bu, 4 dekatlık dinamik aralığı sinir ağı için eşit ağırlıklı hale getiriyor.
  - **Değiştirildi**: `scripts/train_fno.py` — FNO kapasitesi: n_modes 24→32, hidden_channels 64 (değişmedi ama default güncellendi), n_layers 4→5, default epochs 50→80. **CombinedLoss sınıfı**: `LpLoss + h1_weight * H1Loss` (default w=0.3) — H1 gradient loss interference pattern'lerine duyarlı. `--loss` CLI arg (`lp` / `h1` / `lp+h1`). `--no-log-target` flag.
  - **Değiştirildi**: `scripts/train_fno.py` — test sample visualization artık `test_ds.denormalize_target()` ile Pa birimine dönüşüp colorbar'lı çiziliyor. Figürde max pressure değerleri + relative L1% gösteriliyor.
  - **Dokunulmadı**: `simulate.py` veya `phantom.py` — veri pipeline'ı aynı, sadece model tarafı iyileştirildi.

---

- [ml] **İlk preliminary FNO eğitimi tamamlandı.** 200 sample × 50 epoch CUDA üzerinde, sonuç: pipeline çalışıyor ama FNO interference pattern'lerini öğrenemiyor (test LpLoss 2.47).
  - **Çalıştırıldı**: `scripts/train_fno.py --epochs 50 --batch-size 4 --n-modes 20 --hidden-channels 48`
  - **Metrik**: train LpLoss 1.89, val 2.39, test 2.47. Loss eğrisi epoch 25'ten sonra plato — underfit (overfit değil; train-val gap küçük).
  - **Görsel teşhis**: `outputs/fno_run/test_sample.png` — ground truth'ta temiz yatay dalga bantları, FNO prediction anlamsız texture, error map tamamen yüksek. Model dalga envelope'ını bile yakalayamadı.
  - **Kök neden**: (a) 200 sample çok az (Kumar 2024: 1000 %2 hata), (b) n_modes=20 Fourier kapasitesi 316×256 interference'leri yakalamaya yetersiz, (c) target linear MSE/Lp için fazla dinamik aralıklı (9 kPa → 7.6 MPa).
  - **Çıktılar**: `outputs/fno_run/best.pt`, `loss_curve.png`, `test_sample.png`.

---

- [data] **200 sample OpenBreastUS dataset üretildi.** `breast_train_speed.mat`'tan 200 rastgele fantom + rastgele tümör + native path simülasyonlar.
  - **Çalıştırıldı**: `scripts/generate_dataset.py --mat breast_train_speed.mat --n 200 --grid 256 --out data/dataset_v0.h5`
  - **Çıktı**: `data/dataset_v0.h5` (261 MB). Layout: `inputs (200, 3, 316, 256)`, `targets (200, 1, 316, 256)`, `focus (200, 2)`, `phantom_index (200,)`. Attrs: dx=0.2mm, f0=1MHz, source_amp=1MPa, water_standoff=12mm.
  - **İstatistikler**: c mean 1494±21 m/s, rho mean 989±18 kg/m³, alpha mean 0.576±0.29, p_max range 9 kPa..7.6 MPa. 73.6% pixel non-trivial.
  - **Süre**: ~78 dakika CPU (dataset yükleme 50 sn + 200 × ~22 sn/sample padded grid).

---

- [data] **MATLAB v7.3 HDF5 format desteği eklendi.** `breast_train_speed.mat` (1.69 GB) scipy.io.loadmat ile açılamıyordu — HDF5 tabanlı.
  - **Düzeltildi**: `src/phantom.py` — `load_openbreastus_speedmaps()` artık dosya header'ından format algılıyor (`MATLAB 7.3` prefix). v7.3 için h5py backend, eski v7 için scipy.io.loadmat. Her iki format için ayrı ayrı eksen normalizasyonu yapıyor, fantom indeksini her zaman axis 0'a taşıyor (en büyük eksen = N phantoms).
  - **Doğrulandı**: train volume shape `(7200, 480, 480)` olarak normalize edildi. Spot check: phantom 0/100/1000/5000/7199 farklı std değerleri (gerçek farklı fantomlar, tek değerli dilim yok).
  - **Teknik detay**: h5py OpenBreastUS v7.3 dosyasından transpose olmadan `(H, W, N)` döndürüyor — MATLAB column-major saklama artefaktı. Loader `np.argmax(arr.shape)` ile en büyük ekseni N sayıyor, diğer iki ekseni orijinal sırayla koruyor.

---

- [simulation] **Water standoff padding eklendi.** NATIVE path'te kaynak meme deri arayüzüne çok yakın olduğu için enerji yüzeyde kaybediliyordu. Kaynak öncesi su tamponu çözdü.
  - **Değiştirildi**: `src/simulate.py` — `SimConfig.water_standoff_mm=12.0` (default) ve `water_speed=1500.0`. `source_axial_offset` 30→15 (padded domain'in üstünden). `aperture_frac` 0.66→0.80. `t_end` 60µs→80µs (padded grid'de dalganın yolculuğu daha uzun).
  - **Eklendi**: `_pad_water_standoff()` helper — medium haritalarına üstten `pad = round(water_standoff_mm * 1e-3 / dx)` satır su prepend'liyor. Focus koordinatını kaydırıyor. Dönüş dict'ine `pad` alanı eklendi.
  - **Sonuç**: Simülasyon çıktı grid'i `256 → 316 axial` (60 satır su), dalga meme içine düzgün propagate ediyor. Odak hedefe pixel-perfect değil ama çeşitlilik için istenen davranış — model aşırı spesifik odak geometrisine overfit olmuyor.

---

- [dokuman] **Proje dokümantasyonu oluşturuldu.** `system_map.md` ve `Developments.md` LLM context dosyaları yazıldı.
  - **Oluşturuldu**: `system_map.md` — projenin statik anatomisi, tech stack, pipeline, data layer, module haritası.
  - **Oluşturuldu**: `Developments.md` — kronolojik changelog.

---

## 2026-04-08

- [dokuman] **Proje dokümantasyonu oluşturuldu.** `system_map.md` ve `Developments.md` LLM context dosyaları yazıldı.
  - **Oluşturuldu**: `system_map.md` — projenin statik anatomisi, tech stack, pipeline, data layer, module haritası.
  - **Oluşturuldu**: `Developments.md` — kronolojik changelog, bu ilk giriş.

---

- [ml] **NATIVE sound-speed path eklendi (quantization bypass).** OpenBreastUS'un sürekli heterojen ses hızı haritasını bozmadan k-Wave'e yedirmek için ikinci bir entry point hazırlandı. Makale argümanını güçlendiriyor ("OpenBreastUS'un full-resolution heterojenliğini koruyoruz").
  - **Değiştirildi**: `src/simulate.py` — `speed_to_density()` (linear ρ = 0.85·c − 282, Mast 2000 türevi) ve `speed_to_alpha()` (fibroglandüler zirveli Gauss karışımı) helper fonksiyonları eklendi. Ortak `_run_with_medium()` driver'ı çıkartıldı, hem label hem native path buraya gelir. `run_focused_sim_from_speed(speed_map, focus_yx, config)` yeni fonksiyonu eklendi. Eski `run_focused_sim()` artık ortak driver'a delegeliyor.
  - **Değiştirildi**: `src/phantom.py` — `insert_tumor_speed(speed_map, tumor_speed=1550, host_speed_range=(1430, 1560))` continuous speed map üzerinde tümör enjekte eden fonksiyon eklendi. Host tissue detection: diskin %70'i yumuşak doku içindeyse kabul. `load_openbreastus_speed(mat_path, phantom_index, target_size)` sadece ham ses hızı döndüren yalın loader eklendi.
  - **Yeniden yazıldı**: `scripts/run_openbreastus_test.py` — native path'i kullanacak şekilde; 4-panel preview (raw speed | speed+tumor | derived density | peak pressure).

---

- [simulation] **İlk NATIVE path koşusu başarılı.** OpenBreastUS fantom #0 üzerinde native path tek-örnek test.
  - **Düzeltildi**: Quantized path'teki "enerji skin arayüzünde sıkışıyor" problemi çözüldü — dalga artık meme iç dokusuna giriyor.
  - **Metrik**: `p_max` range 17 kPa → 3.08 MPa, CPU run time 4.44 s (256×256 grid).
  - **Çıktı**: `outputs/obus_native_0000.png` + `.npz` — 4-panel önizleme kaydedildi.
  - **Bilinen sorun**: Odak noktası hedefe pixel-perfect oturmuyor (su-meme standoff mesafesi yetersiz), ama dataset üretimi için sorun değil — çeşitlilik sağlıyor.

---

- [pipeline] **Orientasyon + visualizasyon fix.** Sentetik smoke test'in ilk sonucunda kaynak-hedef geometrisi görsel olarak belirsizdi. Eksen konvansiyonu netleştirildi, görselleştirmeye overlay'ler eklendi.
  - **Değiştirildi**: `src/simulate.py` — `SimConfig.source_y_offset` → `source_axial_offset`, `source_axial_offset=30` (önceki 10), `aperture_frac=0.66` yeni parametre. Eksen konvansiyonu dokümante edildi: axis 0 = axial (propagation), axis 1 = lateral. Return dict'e `source_mask` eklendi (plotlama için).
  - **Değiştirildi**: `scripts/run_single_test.py` — mm ekseni (`extent_mm`), magenta source line overlay, cyan target star, percentile clipping (`vmax = 99.5%`), başlıkta peak pressure değeri.
  - **Doğrulandı**: Re-run sonrası 100× odaklanma kazancı, temiz V-pattern, 11.03 s CPU run time. `outputs/sample_0000.png` güncellendi.

---

- [data] **OpenBreastUS veri seti doğrulandı ve loader yazıldı.** arXiv:2507.15035'in HuggingFace'deki dataset'i (`OpenBreastUS/breast`) indirildi ve içeriği inspect edildi.
  - **İndirildi**: `data/breast_test_speed.mat` (195 MB, 800 fantom) — curl ile HF resolve URL'den çekildi. `data/breast_train_speed.mat` (1.69 GB, ~7200 fantom) user tarafından indirildi.
  - **Doğrulandı**: `scipy.io.whosmat` ile — key `'breast_test'`, shape `(800, 480, 480)`, dtype `float32`, range 1403.15 ... 1597.67 m/s, mean 1491.71. Veri zaten 2D — dilim seçimi gereksiz.
  - **Değiştirildi**: `src/phantom.py` — `load_openbreastus_speedmaps()` candidate key listesi gerçek isimlere (`breast_test`, `breast_train`) sabitlendi. `openbreastus_to_label_map()` quantization eşikleri OpenBreastUS değer dağılımına göre recalibrated (bg<1420, fat 1420-1480, fibro 1480-1560, skin>1560). `load_openbreastus_phantom()` refactor — 2D native format için `target_size` parametresi (bilinear resize via `scipy.ndimage.zoom`).
  - **Oluşturuldu**: `scripts/run_openbreastus_test.py` — gerçek fantom üzerinde ilk uçtan uca test (quantized path, sonra NATIVE'e yeniden yazıldı).

---

- [research] **Web search: OpenBreastUS bulundu.** Literatür raporundaki boşluğu doğrudan hedef alan bir dataset keşfedildi.
  - **Kaynak**: Zeng et al., arXiv:2507.15035 (Jul 2025) — "OpenBreastUS: Benchmarking Neural Operators for Wave Imaging Using Breast Ultrasound Computed Tomography"
  - **HF**: `https://huggingface.co/datasets/OpenBreastUS/breast` — CC-BY 4.0, 8000 anatomik meme fantomu, 16M wavefield simülasyonu, 4 yoğunluk sınıfı (HET/FIB/FAT/EXD), frekans aralığı 0.3–0.65 MHz.
  - **Karar**: Dataset'in **fantomları** kullanılacak (breast_train/test_speed.mat), ama **wavefield'ları** değil — çünkü OpenBreastUS USCT görüntüleme için 0.3-0.65 MHz'de yapılmış, bizim hedef HIFU 1 MHz. Aynı fantomları kendi HIFU simülasyonlarımızda kullanarak "OpenBreastUS fantomlarını HIFU rejimine genelleştirme" orijinallik argümanı kurulacak.

---

- [altyapi] **Python bağımlılıkları kuruldu.** `pip install -r requirements.txt` başarıyla tamamlandı.
  - **Oluşturuldu**: `requirements.txt` — numpy≥1.24, scipy≥1.11, matplotlib≥3.7, h5py≥3.9, k-wave-python≥0.3.5, torch≥2.1, neuraloperator≥1.0.
  - **Yüklendi**: `k-wave-python 0.6.1`, `neuraloperator 2.0.0`, `torch`, `scipy 1.15.3`, `tensorly 0.9.0`, `tensorly-torch 0.5.0` ve geçişli bağımlılıklar.
  - **Platform**: Windows 11, Python 3.13, CPU-only (GPU binary sonra eklenecek).

---

- [ml] **Pipeline iskeleti oluşturuldu.** Uçtan uca (phantom → sim → dataset → FNO training) iş akışının tüm modülleri yazıldı.
  - **Oluşturuldu**: `src/tissue_properties.py` — OA-Breast etiket konvansiyonu (0/2/3/4/5/6), Tablo 8 ortalama değerleri, `labels_to_maps()` discrete label → (c, ρ, α) dönüştürücü.
  - **Oluşturuldu**: `src/phantom.py` — `make_synthetic_phantom()` prosedurel fantom, `load_oa_breast_slice()` OA-Breast DAT loader (şimdilik unused, OpenBreastUS yeğlendi), `insert_tumor()` discrete label path tümör enjeksiyonu.
  - **Oluşturuldu**: `src/simulate.py` — `SimConfig` dataclass, `run_focused_sim()` 2D line-source + geometric delays + peak pressure sensor, k-wave-python 0.6.1 API'sine uyumlu.
  - **Oluşturuldu**: `src/dataset.py` — `PressureFieldDataset` HDF5 wrapper, per-channel mean/std normalize (inputs), max-abs scaling (targets).
  - **Oluşturuldu**: `scripts/run_single_test.py` — sentetik smoke test, 3-panel PNG (labels | c | p_max).
  - **Oluşturuldu**: `scripts/generate_dataset.py` — argparse'lı batch generator, `(N, 3, H, W)` input + `(N, 1, H, W)` target HDF5 çıktısı.
  - **Oluşturuldu**: `scripts/train_fno.py` — `neuralop.models.FNO`, AdamW + cosine scheduler, LpLoss train/eval, best checkpoint, test sample visualization + loss curve PNG.
  - **Oluşturuldu**: `README.md` — kurulum, quick start, OpenBreastUS indirme adımları, yol haritası.

---

- [research] **İlk session: proje kapsamı ve literatür.** Kullanıcı mevcut literatür raporunu (30+ çalışma) paylaştı, Gülşah Hoca'nın "AI side literature scan and provide some beneficial stuff" talebi netleştirildi.
  - **İncelendi**: `Literatur_Raporu.pdf` — PINN, CNN/U-Net, Neural Operators (DeepONet/FNO/WNO), GAN/Transformer özetleri. Ana bulgu: literatürün ezici çoğunluğu transkraniyal, yumuşak doku heterojenliği boş.
  - **İncelendi**: `Gorsel1.jpeg` (Tablo 8 — meme dokusu fiziksel parametreleri) ve `Gorsel2.jpeg` (3D Arşimet transdüser + meme fantomu HIFU simülasyonu).
  - **Karar**: Birincil mimari **FNO2d**, ikincil DeepONet. Birincil veri **OpenBreastUS** (sonradan bulundu), yedek **OA-Breast**. 2D'de başla, 3D'yi sonra eklenecek.
