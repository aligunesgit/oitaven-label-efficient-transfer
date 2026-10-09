# İlk transfer karşılaştırması

45 model ve yeniden üretilmiş tahmin/metrikleri doğrulandı. Aktif öğrenme, maliyet katsayısı veya test değerlendirmesi kullanılmadı.700 hedef etiketi CNN seçim/güvenilirlik kontrolü veya doğrudan eğitim için ayrıldı;2000 ayrı geliştirme etiketi değerlendirme maliyetidir.

En iyi macro-F1 ve mIoU,700 hedef etiketiyle doğrudan ince ayarlanan spektral–mekânsal CNN’de. Aynı hedef etiketlerini kullanan Random Forest’a karşı macro-F1 farkı beş seed’de de pozitif. Ancak RF’nin genel doğruluğu daha yüksek. Bu, bütün metriklerde üstünlük değil; nadir sınıf/ortalama sınıf başarısı ile toplam doğruluk arasında bir farktır.

Self-training ve seçici prototip hizalama kaynak CNN’ine göre macro-F1 avantajı göstermedi. Bazı seed’lerde pseudo-label seçim kümesi boş kaldı. Bu koşular source-continued kontrolüyle aynı olabilir; gerçek bir adaptasyon kazancı sayılmaz.

| Seed | Genel pseudo örnek | Filtreli pseudo örnek |
|---|---:|---:|
| seed_42 | 0 | 0 |
| seed_43 | 1200 | 400 |
| seed_44 | 1177 | 400 |
| seed_45 | 0 | 0 |
| seed_46 | 0 | 0 |

Bu sonuçlar yeni yöntem üstünlüğü veya yayınlanabilir özgünlük kanıtı değildir. CNN+700-label supervised adaptation, temel uygulanabilirlik için güçlü bir referans sunar. Tek mekânsal ayrım ve tek sahne kullanıldı. Daha fazla sahne/ayrım, eş bütçeli güçlü modeller ve bağımsız test gerekir.
