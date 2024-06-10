# QA-using-RAG-and-OPENAI

# Giriş:
Merhaba, bu projede büyük dil modellerini ve Retreival Augmented Generation (RAG) tekniğini kullanarak bir e-ticaret sitesi için müşterilerin ürünler hakkında sorular sorup cevap alabileceği bir Soru-Cevap uygulaması oluşturacağız. Ardından basit arayüz ile bu uygulamaya erişip test edeceğiz.

## Proje'nin Amaçları:

1-) Bu proje ile web sitemizde yer alan ürünler ile alakalı olarak müşterilerin sahip oldukları soruları kolayca sorup cevap alabilecekleri bir soru cevap uygulaması geliştireceğiz. Böyle bir uygulamanın müşteriler için pek çok faydası bulunmaktadır. İşte sıralayabileceğimiz bazı faydalar:

* Müşteri Memnuniyetini Artırmak: Müşterilere ürünler hakkında hızlı ve doğru bilgi sağlayarak memnuniyeti artırmak.
* 24/7 Destek: Soru-cevap uygulaması sayesinde müşteriler, mesai saatleri dışında bile sorularına yanıt bulabilirler.
* Detaylı Bilgi Sağlama: Ürünlerle ilgili sıkça sorulan sorular ve cevaplar, potansiyel müşterilere detaylı bilgi sunar.
* Hızlı ve Kolay Erişim: Müşteriler, ürünlerle ilgili sorularına anında cevap alabilirler, bu da müşteri deneyimini olumlu yönde etkiler.
* Geri Bildirim Toplamak: Müşterilerin soruları ve geri bildirimleri sayesinde ürün geliştirme ve iyileştirme süreçlerine katkıda bulunmak.
* İletişim Kolaylığı Sağlamak: Müşterilerin soru sorma sürecini kolaylaştırarak müşteri hizmetlerine olan talebi azaltmak.
* Satışları Artırmak: Ürünler hakkında detaylı bilgi sahibi olan müşterilerin satın alma kararlarını hızlandırmak ve böylece satışları artırmak.
* Tavsiyeler Sunmak: Müşterilerin içinde bulundukları duruma göre (örneğin bir hediye almak sürecinde) onları yönlendirmek ve yardımcı olmak.
* Analitik Veri Sağlama: Müşterilerin en çok hangi konularla ilgilendiğini ve hangi soruları sorduğunu analiz ederek, ürün ve hizmet geliştirme süreçlerine veri sağlama imkanı sunar.


# Detaylı Açıklama:

## 1-) Modelde kullanmak üzere veri setinin hazırlanması:

Bu proje kapsamında veri setimiz sitemizde yer alan ürünleri altında yer alan yorumlar, satıcı-alıcı soru cevapları ve ürünlere ait teknik özelliklerin yer aldığı veriler olacaktır. Bu veriler bir e-ticaret sitesi için oldukça kolay erişilebilecek verilerdir. Fakat bu proje kapsamında ben farklı büyük dil modellerini (Gemini,ChatGPT,Mistral) kullanarak sentetik satıcı-alıcı soru cevap, ürün yorumları ürettim. Bu yorumları pozitif, negatif ve nötr olacak şekilde 3 farklı tonda ürettirdim . Son olarak ürünlerin teknik özelliklerini yer aldığı verileri ise Teknosa, Vatan Bilgisayar gibi teknoloji sitelerinden aldım. Bu yazının bulundupu GitHub Repo'sunda ürettiğim sahte soru-cevap ve kullanıcı yorumlarını "teknik_ozellikler.txt", "yorum_soru_cevap.txt" text dosyaları altında bulabilir ve kendi projeleriniz için kullanabilirsiniz.

## 2-) Proje Aşamaları:

Bu proje için kullanımı ve uygulaması daha kolay olay Retreival Augmented Generation yöntemini tercih ettim. Bunun sebebi bir dil modelini fine-tuning etmenin RAG yöntemine kıyasla çok daha teknik personel ve donanım (hesaplama gücü) gerektirmesidir. Bunun yerine bir dil modelinin çıktısını sürekli olarak güncelleyebileceğimiz bir veritabanı ile desteklemek hem uygulaması hem de gerektirdiği teknik ve hesaplama gücü açısından çok daha uygun bir çözüm olarak gözükmesidir.

Şimdi proje kodlarını açıklayalım.

### 1-) Gerekli Kütüphanelerin İmport Edilmesi:

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/69aa3d20-9a45-4d80-8d2d-21da09a92615)

Proje için gerekli kütüphaneleri import ediyoruz. Bu kütüphanerin nasıl ve ne amaçla kullanıldıklarını ilerleyen bölümlerde anlatacağım.

Şimdi içinde satıcı-alıcı soru cevap ve müşteri yorumlarının yer aldığı metinleri dizinden okuyalım.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/0e63c80f-9b00-45ec-8ea7-729b6065a88f)

Verilerimizi "docs" adlı değişken içinde saklıyoruz. Fakat bir sorun var. Dil Modellerinin belirli bir bağlam penceresi vardır, dolayısıyla tüm verileri tek seferde bir dil modelinin promptuna aktarırsak model buradan hangi soru-cevapların veya yorumların kullanacağını başarılı bir şekilde kestiremez dolayısıyla verilerimizi chunks adı verilen daha küçük parçalara bölüyoruz. Elimizdeki veriler soru-cevap ve yorumlardan oluştuğu için daha küçük bir chunk_size ve chunk_overlap kullanıyorum.

Ardından chunklar haline getirdiğim verileri bir VectorStore'da depolamamız gerekiyor. Bunun için HuggingFace'de yer alan farklı Embeeding modellerini (Setence-Transformer yazarak bulabilirsiniz) kullanabilirsiniz. Ayrıca bu proje kapsamında Türkçe yorumlar ile çalıştığım için farklı Embeeding modellerini denedim ve sonuç olarak en başarılı olanının OpenAI'nın Embeeding modeli olduğuna karar verdim. Bu sebeple OpenAI Embeeding modeli ile devam edeceğiz.

Son olarak VectorStore'a bir soru sorup getirdiği cevapları kontrol ediyoruz. Güzel gözüküyor.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/be6c6a5b-8af6-49a2-b478-8927f66325ea)

Ardından bir Retreiver nesnesi oluşturuyoruz bunu kullandığımız VectorStore'u basitçe bir retreiver'a dönüştürerek yapıyoruz. Son olarak kullanacağımız dil modelini (bu projede GPT-3.5 kullanmılmıştır) oluşturuyoruz.

Şimdi örnek bir prompt oluşturuyoruz, bu promp'da bildiğiniz gibi context kısmına soru ile alaklı VectoreStore'dan getirilen içerikler gelecek. Question kısmında ise kullanıcını sorusu yer alacak. Ardından LangChain'in bize sunduğu Chain yönteminden yararlanarak retreiver'dan getirilen sonucu prompt'un içine alan, ardından bunu LLM'ye aktaran ve son olarak LLM çıktısnı daha anlaşılabilir bir şekilde osn kullanıcıyı gösteren bir OutputParser kullanarak bir chain (zincir) oluşturuyoruz.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/1f931ed5-6c5d-4dc8-bd92-ccafd2e7ced4)

Bir örnek prompt girelim ve sonuçları görelim.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/86b11546-0a24-47a1-b762-3c3a6a350c0f)

Veri setini ben hazırladığım için böyle bir bilginin soru-cevap kısmında var olduğunu biliyorum, isterseniz bu chain'in ayrıca bu çıktıyı verirken kullandığı kaynakları döndürmnesini de isteyebilirsiniz.

Fakat burada bir sorunumuz daha var. Bu sohbet robotu geçmişi hatırlayamıyor. Bir örnek üzerinden açıklayayım.

Mesela sohbet robotuna "İphone 14'ün kaç megapiksellik bir kamerası vardır" diye bir soru sorduğunuzda model "20 megapiksel" diye bir yanıt döndürdüğünü varsayalım. Ardından "peki bu telefonun fiyatı nedir?" gibi bir soru sorduğunuz zaman model bu sorunun bir önceki soru ile bağlantılı bir soru olduğunu anlayamaycaktır. Dolayısıyla bu modele eski mesajları (sohbeti) hatırlayabilmesi için bir chat_history ve kendisine gelen soruyu retreiver üzerinde doğru bir şekilde arama yapabilmesi için yeniden dizayn eden bir LLM daha gerekli. Yani az önce sorulan devam sorusunu (peki bu telefonun fiyatı nedir?) alıp (İphone 14'ün fiyatı nedir?) şeklinde düzenlememiz lazım.

İlk önce LLM'ye devam sorularını retreiver üzerinde arama yapabilmek için bağlamsallaştırmasını gerektiğini belirten bir prompt oluşturalım.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/248ef05f-c3cc-4f14-90b4-088514e2f82e)

Ardından içinde hem yukarıda belirtilen "system prompt" hem geçmiş sohbeti barındıran "chat history" hem de kullanıcını sorusunu barındıran bir prompt oluşturalım.


![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/82b188e0-f8f2-456d-bfb1-24764f5b509e)




