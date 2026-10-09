# İlk pilot değerlendirmesi

Bu çalışma yalnızca geliştirme bölgelerinde yapılmış bir fizibilite pilotudur. Orijinal test blokları ve test örnek dosyaları byte düzeyinde değişmedi; test değerlendirilmedi. Beş seed aynı mekânsal ayrımı paylaşır.

İlk geometrik pilot iki sınıflı değerlendirme desteği nedeniyle reddedildi. NumPy2/Accelerate sayısal uyarıları üreten tekrar da reddedildi. Kabul edilen koşu ayrı NumPy1.26.4 ortamında, RuntimeWarning hata sayılarak tamamlandı. Kontroller200 sonucu, modelden yeniden hesaplanan olasılıkları, metrikleri, oracle etiketlerini ve aynı başlangıç ID’lerini doğruladı.

Yeni ayrımda test dışındaki256px bloklar32px tamponla kullanılır. Kontrol/geliştirme sınıf desteği için test dışı benchmark etiketlerinden faydalanıldığı açıktır; bu gerçek bir etiketsiz saha protokolü değildir. Yöntem sorguları edinilmemiş etiketlere erişmez. Geliştirme örnekleme zenginleştirmesi, sınıf seçilme olasılığı ağırlıklarıyla düzeltilir.

1046 edinilmiş hedef etiketi +700 kontrol/eş bütçeli eğitim etiketi kullanıldı.2000 geliştirme etiketi ek değerlendirme maliyetidir. Kaynak eğitim etiketleri ayrıca hazırlanmış manifestte sayılır.

| Yöntem | Macro-F1 | Sabit senaryo maliyeti (düşük iyi) | Doğruluk |
|---|---:|---:|---:|
| source_only | 0.3809 | 0.5460 | 0.6539 |
| coral_unlabeled | 0.3273 | 0.8623 | 0.3827 |
| adaptive_entropy | 0.3676 | 0.7699 | 0.4335 |
| adaptive_risk | 0.3890 | 0.6646 | 0.4509 |
| budgetmatched_entropy | 0.3719 | 0.7122 | 0.3630 |
| budgetmatched_risk | 0.3408 | 0.7965 | 0.2641 |
| fixed_entropy | 0.2840 | 0.9240 | 0.2745 |
| fixed_random | 0.3378 | 0.7456 | 0.3565 |
| fixed_risk1 | 0.3405 | 0.7615 | 0.3655 |
| fixed_risk3 | 0.3001 | 0.8455 | 0.2472 |
| fixed_risk5 | 0.2836 | 0.8725 | 0.2131 |
| fixed_risk7 | 0.2638 | 0.8905 | 0.1948 |

Yorum: CORAL bu kaynak/aday havuzu ve özelliklerde negatif transfer gösteriyor. Adaptif risk politikasının bazı sabit politikalara göre avantajı, kaynak modelinin maliyetini iyileştirdiği anlamına gelmiyor. Maliyetler varsayımsaldır; gerçek çevresel zarar ölçülmedi.

Sonraki araştırma kapısı: sınıf-dengeli kaynak örneklemesinin hizalamayı nasıl etkilediğini, olasılıkların kalibrasyonunu ve güçlü doğrusal olmayan kaynak baseline’ını incelemek. Mevcut pilotla yöntem üstünlüğü veya etiket tasarrufu iddiası kurulmamalı.
