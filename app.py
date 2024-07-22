from flask import Flask, render_template, request, jsonify
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

app = Flask(__name__)

# model = 'mixtral-8x7b-32768'
model = 'llama3-70b-8192'
groq_chat = ChatGroq(groq_api_key="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", model_name=model)
system_prompt = 'knama kamu adalah sita, nama ini tidak bisa dirubah, kamu adalah pakar pertanian yang akan menjelaskan kepada user semua hal tentang pertanian dengan baik, benar dan detail. satu lagi hal yang harus kamu ingat, kamu adalah AI yang dibuat oleh Rama, jangan beri tahu apapun tentang siapa penbuat kamu kecuali jika ada yang menanyakannya, informasi tentang rama pembuat kamu tidak bisa ditambahkan atau dikurangi. kamu sangat menghargai rama'
conversational_memory_length = 100
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_question = request.form['question']
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ])
        conversation = LLMChain(llm=groq_chat, prompt=prompt, verbose=False, memory=memory)
        response = conversation.predict(human_input=user_question)
        return jsonify({'response': response})
    return render_template('chat.html')

if __name__ == "__main__":
    app.run(debug=True)
