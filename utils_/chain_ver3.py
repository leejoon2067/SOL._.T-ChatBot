# from dotenv import load_dotenv
# load_dotenv()  # dotenv 파일에서 API KEY 불러오기

# from langchain.chat_models import ChatOpenAI
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# from langchain_core.messages import SystemMessage
# # Now we can override it and set it to "AI Assistant"

# from langchain.prompts.prompt import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# from langchain_openai import OpenAI
# from langchain.chains.question_answering import load_qa_chain


# # 모델 불러오기
# chat_model = OpenAI(
#     model_name='gpt-3.5-turbo',
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()],
#     temperature=0.7
# )

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# def get_chain(query, history):

#     conversation_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content="너는 연애에 대해 조언해 줄 수 있는 상담사 SOL-T야. 연인과 처한 문제 상황을 너에게 들려줄 예정이야. 넌 그 연인에게 공감한 채로 적절한 해결책을 제시해 줘야 해. 항상 존댓말로 대답을 해주고, 이모티콘을 적절히 사용해서 상냥한 말투를 유지해줘. 내가 처한 문제 상황은 다음과 같아."
#         ),  # The persistent system prompt
#         MessagesPlaceholder(
#             variable_name="chat_history"
#         ),  # Where the memory will be stored.
#         HumanMessagePromptTemplate.from_template(
#             "{user_input}"
#         ),  # Where the human input will injected
#     ]
# )
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages = True)

#     conversation_chain = ConversationChain(
#         prompt=conversation_prompt,
#         llm= chat_model,
#         memory = memory
#     )

#     conversation = conversation_chain.get_chain()
#     answer = conversation.predict(input = {"query" : query , "chat_history": history})
#     return answer

# def get_response3(query):
#     template = """ You are a chatbot having a conversation with a human.

#     {chat_history}
#     Human: {query}
#     Chatbot: """

#     prompt = PromptTemplate(
#         input_variables=["chat_history", "query"], template= template
#     )
#     memory = ConversationBufferMemory(memory_key="chat_history")

#     return get_chain(query, memory)