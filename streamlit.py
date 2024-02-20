import streamlit as st
from utils import rag_func
from utils import s
from utils import process_llm_response
if "messages" not in st.session_state.keys():
    st.session_state["messages"]=[
        {"role":"assistant","content": "Hello there,how can I help you today."}
    ]

#display the messages
if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({
        "role":"user",
        "content": user_prompt})
    

    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"]!= "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking...."):
            ai_response = rag_func(user_prompt)
            llm_response = process_llm_response(ai_response)
            res = s(ai_response)
            st.write(llm_response)
            st.write(res)


    new_ai_message = {"role": "assistant", "content": llm_response}
    st.session_state.messages.append(new_ai_message)



