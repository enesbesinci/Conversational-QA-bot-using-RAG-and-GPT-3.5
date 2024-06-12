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

Then we need to store the chunked data in a VectorStore. To do this, we need a VectorStore and an embedding model to embed the chunks. There are several embeeding models available in HuggingFace (you can search for them by typing Setence-Transformer). But since we are working with Turkish data in this project, we will continue with OpenAI's embeeding model. The reason for this is that I tried different embeeding models in HuggingFace, but I got the most successful result with OpenAI's embeeding model. We will also be using Meta's VectorStore, FAISS (Facebook AI Similarity Search).

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/969d6396-5508-4c6d-81d9-03579f2192d8)

Finally, we ask VectorStore a question and check the answers. The results look good.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/5b922553-ef2f-44cf-9064-dee86486896b)

Then we create a Retreiver object that takes a user's question and searches and returns the content related to the question in the VectorStore. We do this by simply converting the VectorStore we use into a Retreiver. Finally, we create the language model we will use (we used GPT-3.5 in this project).
Note: We set the Temperature parameter to 0.1, which will perform better for this type of question-answer application. With this parameter, we set the model answers to be more deterministic.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/c8e96918-505b-45b3-a682-bec943485f5b)


Now we are creating a sample prompt. The ‘Question’ section will contain the user's question. In the ‘context’ section, the content taken from VectoreStore related to the question will come. Then we will bring our components together using the ‘LCEL Runnable’ protocol LangChain offers us. Thus, we will have a chain that takes the content from the Retreiver into the prompt, then passes it to the LLM with the question, and finally an OutputParser that shows the LLM output to the end user in a more understandable way.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/bf1cb90d-2a42-4db3-8702-92f11307b7d8)

With an example prompt and let's see the results.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/25a43424-3470-4442-b14e-219ba72e7cf5)

As you know, since I prepared the dataset, I know that the answer is correct (our fictitious business makes 12 instalments to credit card).

But we have a problem here. This chatbot cannot remember the past. Let me explain with an example.

For example, if you ask the chatbot a question like "How many megapixels does the iPhone 14 camera have?", the model will say "20 megapixels". If you then ask the chatbot a follow-up question like "So what's the price of this phone?", the model won't realise that the follow-up question is related to the previous question.

So we need a chat_history to remember old messages (chat), and an LLM to reformulate the follow-up question so that it can correctly search for the follow-up question. So we need to take the follow-up question (What is the price of this phone?) and rephrase it as (What is the price of the iPhone 14?). Note that the retreiver will return bad respones for the first question.

First, let's create a prompt telling the LLM that it needs to contextualize the follow-up questions in order to search them on the retriever.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/275d2da8-4df4-4610-93db-aeaf130414d8)

Then, let's create a prompt that contains both the "system prompt" mentioned above, the "chat history" containing the past chat, and the user's question, and combine this prompt with the LLM using the create_stuff_documents_chain function.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/aac659ca-9b2f-40a1-8309-c0c1c7ac49fe)

Now let's put all these functions we created together.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/fefdb071-082b-42bc-b028-3ff10b353835)

Now let's ask a few sample questions and see the model's response.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/c8722d18-c7b7-41fb-9d4b-3b5387b3009a)

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/d5d34797-b3f1-4635-9a55-bfaae9faeca3)

As you can see, in the second question, the model realized that the question was related to the first question, rearranged the question and then answered it.

Now, let's create a simple interface with Gradio for this application, ask questions and see the answers.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/c7eaa189-21cf-43bb-8b2f-c1f56b8cac11)

We created the simple interface. You should be able to access the interface using the link that shown below the code line.

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/2c697a78-ec67-475e-85e9-9969af2d62d6)

![image](https://github.com/enesbesinci/QA-using-RAG-and-OPENAI/assets/110482608/f096c16c-7d11-4543-b91c-e9750935d958)

As you see in the picture. The model understood the (relevance?) between the first question and the second question. Let's ask another question.

![image](https://github.com/enesbesinci/Conversational-QA-bot-using-RAG-and-OPENAI/assets/110482608/1e332611-5d74-46e1-ae3a-d7ac6cacec6a)

The results look pretty good. Of course, an e-commerce site's data and more advanced models (especially those that perform better in Turkish, such as fine-tuning with Turkish data) may perform better. On the other hand, it is important to note that there are different approaches and methods in applying the RAG method. For example, using different search algorithms such as BS-25 together with semantic search while searching on the retriever can improve the performance of the model. However, since this project is an introductory example, we will not go into details.

## Conclusion:

This project has successfully achieved its goal of developing a Q&A application that can answer customers' questions about products quickly and accurately on e-commerce sites. This system, created using the Retrieval Augmented Generation (RAG) technique, offers many benefits such as providing 24/7 support, providing detailed information, collecting feedback and increasing sales, as well as increasing customer satisfaction. Thanks to the application's simple interface, users can get immediate answers to their questions. This project can be further improved with high quality data and advanced methods, but even in its current form it has the potential to positively impact the customer experience.





















