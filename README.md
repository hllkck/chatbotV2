📚 Dinamik RAG Destekli Dil Eğitmeni Chatbotu (V2)

🌟 Proje Hakkında
Proje, LangChain Expression Language (LCEL) mimarisini kullanarak oluşturduğum, dil öğrenimine odaklanan akıllı bir çeviri botudur. Geleneksel çeviri uygulamalarından farklı olarak, özel bir kelime dağarcığı verisetinden (ChromaDB'de vektörleştirdiğim) seviye bilgisi (A1, B2 vb.) içeren zenginleştirilmiş yanıtlar üretiyor.
(V1 Bilgisi): Projenin ilk sürümü, temel RAG yapısını kullanarak sadece kelime sorgularını yanıtlamaktaydı. V1 sürümünün kod yapısını ve detaylarını [[İLK VERSİYON GİTHUB LİNKİ](https://github.com/hllkck/chatbot)] adresinden inceleyebilirsiniz.
V2 sürümünü, verimlilik ve kullanıcı deneyimini maksimuma çıkarmak için dinamik sorgu yönlendirmesi ve sesli okuma özellikleriyle geliştirdim.

🚀 V2 Mimarisi ve Temel Yenilikler
Proje, gelen sorgunun tipine göre LLM kaynaklarını akıllıca yönetebilen karma mimarisine sahiptir.
1. 🎯 Akıllı Sorgu Yönlendirme (LCEL RunnableBranch)
Bu mimari, kaynak tüketimini optimize ediyor ve kullanıcıya her zaman en hızlı ve doğru yanıtı sunuyor:
Sorgu Tipi	Mekanizma	Faydası
Kelime Anlamı / Kısa Sorgular	Özel Veriseti RAG Zinciri	Verisetimdeki Level (A1, B2) bilgisini mutlak suretle yanıta dahil ediyor.
Cümle Çevirisi / Uzun Genel Sorular	Doğrudan LLM Çağrısı	RAG adımlarını atlayarak gecikmeyi azaltıyor ve hızlı, doğrudan çeviri sağlıyor.

2. 🗣️ Sesli Okuma Entegrasyonu (TTS)
Öğrenme deneyimini zenginleştirmek amacıyla, asistanın ürettiği tüm İngilizce çıktıları otomatik olarak sese dönüştürülüyor ve arayüze bir medya oynatıcı olarak ekleniyor.
•	Teknoloji: gTTS kütüphanesi ve Streamlit'in HTML embed özelliği kullanılıyor.
•	Fonksiyon: extract_english_word ile model çıktısındaki okunması gereken İngilizce kısımlar hassas bir şekilde ayrıştırılıyor.

3. 🛡️ Güvenli ve Merkezi Veri Yönetimi
•	Eski Yöntem: Yerel words.txt dosyasıydı.
•	Yeni Yöntem: Verilerim, Streamlit Cloud ortamında güvenlik ve erişim kolaylığı sağlayan Streamlit Secrets (st.secrets["data_storage"]) üzerinden yükleniyor.

4. 📈 Gelişmiş Bağlam Formatlama
RAG'den gelen verinin LLM tarafından doğru yorumlanmasını sağladım.
•	format_context_with_level fonksiyonu, verisetinden gelen seviye bilgisini (örn: dog A2) yakalıyor ve LLM'in açıkça anlayabileceği formatta sunuyor: [DATASET WORD] dog | Level: A2.

⚠️ Kritik Dikkat Edilmesi Gereken Nokta: Kota Tüketimi
Projeme dahil ettiğim MultiQueryRetriever özelliği, arama kalitesini artırmak için her kullanıcı sorgusunu arkada 3-5 farklı sorguya dönüştürüyor.
Bu, herhangi bir LLM sağlayıcısında (OpenAI, Gemini vb.) API çağrısı tüketimini 4-6 kat artırıyor.
Tavsiye: Uygulamanız yüksek trafik alıyorsa veya kota kısıtlı bir anahtar kullanıyorsanız, MultiQueryRetriever yerine basit _vectorstore.as_retriever kullanılarak LLM çağrısı sayısı kullanıcı başına 1'e düşürülmelidir.

🔒 Güvenlik ve API Kota Koruması
API tüketimini optimize etmek ve uygulamayı kötüye kullanıma karşı korumak için iki temel güvenlik önlemi entegre edilmiştir.
1. ⏱️ Hız Sınırlama (Rate Limiting)
•	Amaç: API çağrısı maliyetini kontrol altında tutmak ve sunucu kaynaklarının aşırı yüklenmesini önlemek.
•	Mekanizma: Kullanıcının arka arkaya çok hızlı sorgu göndermesi engellenir. Her sorgu arasında zorunlu bir bekleme süresi uygulanır.
2. 🎯 Konu Dışı Sorgu Engelleme (OffTopicPrompts Emülasyonu)
•	Amaç: LLM'in yalnızca dil öğrenimi ve çeviri görevlerine odaklanmasını sağlamak ve genel sorular için API çağrısı yapılmasını engellemek.
•	Mekanizma: Kullanıcı girdisi, çeviri veya kelime anlamı dışındaki genel konuları (tarih, siyaset, yemek tarifi vb.) içeriyorsa, sorgu LLM'e gönderilmeden engellenir ve bir uyarı mesajı gösterilir.

⚙️ Ön Gereksinimler
•	Python 3.11
•	OpenAI veya Google Gemini API anahtarı
•	Opsiyonel: Hugging Face API Token 

🚀 Demo
Projeyi canlı olarak deneyin: [[Dil Eğitmeni Chatbotu](https://chatbot-v2-0.streamlit.app/)]
