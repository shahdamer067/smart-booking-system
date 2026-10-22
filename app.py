"""Smart Booking System: booking form + RAG support chatbot + trainable intent classifier."""
import os
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import booking_logic as bl
import intent_classifier as ic

load_dotenv()
st.set_page_config(page_title="Smart Booking System", layout="wide")
st.title("Smart Booking System")

# ---------- API keys: from .env locally, or Streamlit secrets when deployed ----------
for key in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
    if not os.getenv(key):
        try:
            os.environ[key] = st.secrets[key]
        except Exception:
            pass
KEYS_OK = bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GROQ_API_KEY"))


@st.cache_resource
def get_rag_chain():
    from rag import build_chain  # imported here so the booking tabs work even without keys
    return build_chain()


# ---------- Session state ----------
st.session_state.setdefault("training_data", list(ic.DEFAULT_EXAMPLES))
st.session_state.setdefault("model", None)
st.session_state.setdefault("vectorizer", None)
st.session_state.setdefault("suggested_service", None)
st.session_state.setdefault("messages", [{"role": "assistant", "content": "Hello! Ask me about our services, hours or policies."}])

tab_book, tab_chat, tab_agent = st.tabs(["Booking", "AI Chatbot", "Train Agent"])

# ---------- TAB 1: Booking ----------
with tab_book:
    st.header("Book your appointment")
    if st.session_state.suggested_service:
        st.info(f"The agent suggested: **{st.session_state.suggested_service}** (pre-selected below).")

    default_index = (
        bl.SERVICES.index(st.session_state.suggested_service)
        if st.session_state.suggested_service in bl.SERVICES else 0
    )
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Your name")
        service = st.selectbox("Service", bl.SERVICES, index=default_index)
    with col2:
        day = st.date_input("Date", min_value=date.today())
        free = [h for h in bl.slots_for_day(day) if h not in bl.taken_hours(day)]
        if not bl.slots_for_day(day):
            st.warning("We are closed on Sundays.")
        elif not free:
            st.warning("No free slots on this day.")
        hour = st.selectbox("Time", free, format_func=bl.slot_label, disabled=not free)

    if st.button("Confirm booking", disabled=not free):
        errors = bl.validate_booking(name, service, day, hour)
        if errors:
            for e in errors:
                st.error(e)
        else:
            booking_id = bl.save_booking(name, service, day, hour)
            st.success(f"Booking #{booking_id} confirmed for {name.strip()}: {service} on {day} at {bl.slot_label(hour)}.")

    st.subheader("All bookings")
    rows = bl.list_bookings()
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        cancel_id = st.selectbox("Cancel a booking", [r["ID"] for r in rows])
        if st.button("Cancel selected booking"):
            bl.cancel_booking(cancel_id)
            st.rerun()
    else:
        st.caption("No bookings yet.")

# ---------- TAB 2: RAG chatbot ----------
with tab_chat:
    st.header("Chat with support")
    if not KEYS_OK:
        st.warning("Add GOOGLE_API_KEY and GROQ_API_KEY to your .env file to enable the chatbot (see README).")
    else:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if question := st.chat_input("Ask about services, hours, policies..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    answer = get_rag_chain().invoke(question)
                except Exception as e:
                    answer = f"Sorry, the AI service returned an error: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------- TAB 3: Trainable agent ----------
with tab_agent:
    st.header("Train your personal agent")
    st.markdown("Teach a small neural network which service matches what a person says.")
    left, right = st.columns(2)

    with left:
        st.subheader("1. Training examples")
        st.dataframe(pd.DataFrame(st.session_state.training_data, columns=["User says", "Service"]), height=220, hide_index=True)
        with st.form("add_example", clear_on_submit=True):
            new_text = st.text_input("User says:")
            new_label = st.selectbox("Correct service:", bl.SERVICES)
            if st.form_submit_button("Add example"):
                if new_text.strip():
                    st.session_state.training_data.append((new_text.strip(), new_label))
                    st.rerun()
                else:
                    st.error("The example text cannot be empty.")
        if st.button("Train neural network"):
            try:
                st.session_state.model, st.session_state.vectorizer = ic.train(st.session_state.training_data)
                st.success(f"Trained on {len(st.session_state.training_data)} examples.")
            except ValueError as e:
                st.error(str(e))

    with right:
        st.subheader("2. Test the agent")
        text = st.text_input("Describe your need (e.g. 'I feel sick'):")
        if st.button("Predict service"):
            if st.session_state.model is None:
                st.warning("Train the network first.")
            elif not text.strip():
                st.warning("Type something first.")
            else:
                service, conf = ic.predict(st.session_state.model, st.session_state.vectorizer, text)
                if service is None:
                    st.session_state.suggested_service = None
                    st.warning(f"I'm not sure (confidence {conf:.0%}). Try rewording, or add more examples.")
                else:
                    st.session_state.suggested_service = service
                    st.success(f"Suggested service: **{service}** (confidence {conf:.0%}). Open the Booking tab to continue.")
