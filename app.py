import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from utils_.utils import print_messages
from utils_.chain_v1 import get_response
# from utils_.chain_ver2 import get_response2
# from utils_.chain_ver3 import get_response3

st.set_page_config(
    page_title = "연애 솔루션 챗봇 'Sol-T'",
    page_icon = "💞"
)
st.title("당신의 연애 솔루션 챗봇 SOL._.T💞")
st.divider()

st.markdown("##### **😁 쏠티**에게 당신의 고민을 털어보아요!!!")

# API key 설정
load_dotenv()
os.environ.get("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록 출력해주는 함수. 
print_messages()

if user_input := st.chat_input("어떤 것이 궁금하신가요?"):
    # 사용자 입력 
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role = "user", content = user_input))

    # LLM 답변 생성 예시 : ver1 - get_response1
    response = get_response(user_input)
    msg = response

    # LLM 답변 생성 예시 : ver2 - get_response2
    # response = get_response(user_input)
    # msg2 = response

    # LLM 답변 생성 예시 : ver3 - get response3
    # response = get_response3(user_input)
    # msg3 = response

    # AI 답변    
    with st.chat_message("assistant"):
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role = "assistant", content = msg))