import streamlit as st

from litellmchat_streamlit.agent import Agent


def main() -> None:
    st.set_page_config(page_title="LiteLLM Chat Assistant", page_icon="⚡")
    st.title("LiteLLM Chat Assistant ⚡")

    if "agent" not in st.session_state:
        st.session_state["agent"] = Agent()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type a message...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = st.session_state["agent"].chat(prompt)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()