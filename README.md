# Creating a Conversational Question-Answering Bot Using LLM and RAG Technique for an E-Commerce Website:

## Introduction:
Hello everyone, in this project, we will create a Q&A application for an e-commerce website where customers can ask questions about the products and get answers using Large Language Models and Retreival Augmented Generation (RAG) technique. Then we will access and test this application with a simple interface.

### Note: The objective of this project is to illustrate one of the use cases of the RAG technique, so it assumes that you have a basic knowledge of LLMs and the RAG technique.

## Goals of Project:

With this project, we will develop a question-answering application where customers can easily ask questions about the products on our website and get answers. This application has many benefits for customers. Here are some benefits we can list:
* Increasing Customer Satisfaction: To increase satisfaction by providing customers with fast and accurate information about products.

* Providing Detailed Information: Frequently asked questions and answers about products provide detailed information to potential customers.
  
* Quick and Easy Access: Customers can get instant answers to their questions about products, which positively affects the customer experience.

* Collecting Feedback: Contributing to product development and improvement processes through customers' questions and feedback.

* Providing Ease of Communication: Reducing the demand for customer service by facilitating the process of customers asking questions.

* Increasing Sales: To accelerate the purchasing decisions of customers who have detailed information about the products and thus increase sales.

* Providing Recommendations: Guiding and helping customers according to the situation they are in (for example, in the process of buying a gift).

* Providing Analytical Data: It provides the opportunity to provide data to product and service development processes by analysing which topics customers are most interested in and which questions they ask.

* 24/7 Support: Through the question-answering application, customers can find answers to their questions even outside office hours.

Now that you've got the concept, let's write the code to build it.


### Dataset we will be using:

The dataset we will be using within the scope of this project will be the data containing the comments made about the products sold in our e-commerce website, seller-buyer question-answers and technical specifications of the products. These data are quite easily accessible data for an e-commerce website. However, within the scope of this project, I generated synthetic seller-buyer question-answers and product comments using different large language models (Gemini, ChatGPT, Mistral). I generated these comments in 3 different tones: positive, negative and neutral. Then, I took the data containing the technical specifications of the products from technology sites such as Teknosa and Vatan Computer. In the GitHub Repo where this article is located, you can find the fake Q&A conversations and user comments I generated under the text files 'teknik_ozellikler.txt' and 'yorum_soru_cevap.txt' and use them for your own projects.

### Coding:

Firstly, we import the necessary libraries for the project. We will show how and for what purpose these libraries are used in the following sections.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/af204185-950f-4c19-a330-97fb3bcb8ecf)

Now let's read the texts docs from the index, in which there are seller-buyer questions and answers and customer comments and technical properties of the products we sold in our website.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/c402608b-2283-4427-ac40-e62872acdfb4)

We have stored our data in a variable called ‘docs’, but there is a problem. Language Models have a limited context window, so if we pass all the data to the prompt of a language model at once, the model cannot successfully understand which context (questions-answers or comments) to use for the question, so we split our text data into smaller pieces called chunks. I use a smaller chunk_size and chunk_overlap because our data consists of small chunks of text such as questions, answers and comments.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/3e213ff1-8e90-4e12-93a3-0ea2899ec8c5)

Then we need to store the chunked data in a VectorStore. For this we need a VectorStore and an Embeeding model to embed the chunks. For this, you can use different Embeeding models in HuggingFace (you can search by typing Setence-Transformer). But since we are working with Turkish data in this project, we will continue with OpenAI's Embeeding model. The reason for this is that I tried different Embeeding models in HuggingFace, but I got the most successful result with OpenAI's Embeeding model.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/969d6396-5508-4c6d-81d9-03579f2192d8)

Finally, we ask VectorStore a question and check the answers. The results look good.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/5b922553-ef2f-44cf-9064-dee86486896b)

Ardından bir Retreiver nesnesi oluşturuyoruz bunu kullandığımız VectorStore'u basitçe bir retreiver'a dönüştürerek yapıyoruz. Son olarak kullanacağımız dil modelini (bu projede GPT-3.5 kullanmılmıştır) oluşturuyoruz. (Temperature parameteresini modelin çıktılarını daha deterministik yapmak adına 0.1 gibi düşük bir değere ayarladım)

Şimdi örnek bir prompt oluşturuyoruz, bu promp'da bildiğiniz gibi context kısmına soru ile alaklı VectoreStore'dan getirilen içerikler gelecek. Question kısmında ise kullanıcını sorusu yer alacak. Ardından LangChain'in bize sunduğu Chain yönteminden yararlanarak retreiver'dan getirilen sonucu prompt'un içine alan, ardından bunu LLM'ye aktaran ve son olarak LLM çıktısnı daha anlaşılabilir bir şekilde osn kullanıcıyı gösteren bir OutputParser kullanarak bir chain (zincir) oluşturuyoruz.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/1f931ed5-6c5d-4dc8-bd92-ccafd2e7ced4)

Bir örnek prompt girelim ve sonuçları görelim.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/86b11546-0a24-47a1-b762-3c3a6a350c0f)

Veri setini ben hazırladığım için böyle bir bilginin soru-cevap kısmında var olduğunu biliyorum, isterseniz bu chain'in ayrıca bu çıktıyı verirken kullandığı kaynakları döndürmnesini de isteyebilirsiniz.

Fakat burada bir sorunumuz daha var. Bu sohbet robotu geçmişi hatırlayamıyor. Bir örnek üzerinden açıklayayım.

Mesela sohbet robotuna "İphone 14'ün kaç megapiksellik bir kamerası vardır" diye bir soru sorduğunuzda model "20 megapiksel" diye bir yanıt döndürdüğünü varsayalım. Ardından "peki bu telefonun fiyatı nedir?" gibi bir soru sorduğunuz zaman model bu sorunun bir önceki soru ile bağlantılı bir soru olduğunu anlayamaycaktır. Dolayısıyla bu modele eski mesajları (sohbeti) hatırlayabilmesi için bir chat_history ve kendisine gelen soruyu retreiver üzerinde doğru bir şekilde arama yapabilmesi için yeniden dizayn eden bir LLM daha gerekli. Yani az önce sorulan devam sorusunu (peki bu telefonun fiyatı nedir?) alıp (İphone 14'ün fiyatı nedir?) şeklinde düzenlememiz lazım.

İlk önce LLM'ye devam sorularını retreiver üzerinde arama yapabilmek için bağlamsallaştırmasını gerektiğini belirten bir prompt oluşturalım.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/248ef05f-c3cc-4f14-90b4-088514e2f82e)

Ardından içinde hem yukarıda belirtilen "system prompt" hem geçmiş sohbeti barındıran "chat history" hem de kullanıcını sorusunu barındıran bir prompt oluşturalım ve bu prompt ile LLM'i create_stuff_documents_chain fonksiyonun kullanarak birleştirelim.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/a170414e-dd8f-487b-a476-20b5247dffb8)

Şimdi tüm bu oluşturduğumuz fonksiyonları bir araya getirelim.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/b92eb5ef-7207-4e9f-b422-bbab5c791962)

Şimdi bir kaç örnek soru soralım ve modelin cevaplarını görelim.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/14a3d667-b947-434c-bcd2-93f6976b1506)

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/08fa5558-e5bc-466c-b078-5433a0ee7ed3)

Gördüğünüz gibi model ikinci soruda sorunun ilk soru ile bağlantılı olduğunu anladı, soruyu tekrar düzenledi ve tüm adımlarını gerçekleştirerek soruyu cevapladı.

Şimdi bu uygulama için Gradfio ile basit bir arayüz oluşturup tekrar sorular soralım ve cevapları görelim.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/2c697a78-ec67-475e-85e9-9969af2d62d6)

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/f096c16c-7d11-4543-b91c-e9750935d958)

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/74906675-6587-453b-bb29-46ecf97d5604)

Sonuçlar oldukça iyi gözüküyor. Elbete bir e-ticaret web sitesinin sahip olduğu veriler ve daha gelişmiş özellikle de Türkçe dilinde daha iyi performans gösteren, örneğin Türkçe veriler ile fine-tuning edilmiş bir model farklı bir performans gösterebilir. Fakat bu tür bir uygulamanın yüksek miktarda kaliteli veri ve daha gelişmiş yöntemlerle kullanıcı memnuniyetini ve tüketim alışkanlıklarını geliştirip değiştireceği açıktır. Bunlara ek olarak RAG yönteminde farklı yaklaşımlar ve yöntemler olduğunu da belirtmek önemlidir. Mesela retreiver üzerinde arama yaparken semantik arama ile birlikte BS-25 türü farklı arama algoritmalalarını kullanmak modelin performansını artırabilir. Fakat bu proje bir giriş mahiyetinde bir örnek proje olduğu için çok detaylara girmedim.

## Sonuç:

Bu proje, e-ticaret sitelerinde müşterilerin ürünlerle ilgili sorularını hızlı ve doğru bir şekilde cevaplayabilen bir Soru-Cevap uygulaması geliştirme amacına başarılı bir şekilde ulaşmıştır. Retrieval Augmented Generation (RAG) tekniğini kullanarak oluşturulan bu sistem, müşteri memnuniyetini artırmanın yanı sıra 24/7 destek sağlama, detaylı bilgi sunma, geri bildirim toplama ve satışları artırma gibi pek çok fayda sunmaktadır. Uygulamanın basit arayüzü sayesinde kullanıcılar, sorularına anında yanıt alabilirler. Bu proje, yüksek kaliteli veri ve gelişmiş yöntemlerle daha da iyileştirilebilir, ancak mevcut haliyle bile müşteri deneyimini olumlu yönde etkileme potansiyeline sahiptir.






















