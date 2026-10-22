# Smart Booking System

A Streamlit appointment-booking app that combines three components:

1. **Booking form** with real rules: opening hours, closed Sundays, a 30-day booking window, no double-booking. Bookings are stored in SQLite and can be cancelled.
2. **RAG support chatbot** that answers questions about services, hours and policies from a knowledge base (`data/booking_docs.txt`).
3. **Trainable intent classifier** (a small neural network) that maps what a person says ("my back hurts") to a service ("Massage") and pre-selects it in the booking form.

## How it works

```
data/booking_docs.txt -> text chunks -> Gemini embeddings -> FAISS vector store
                                                                  |
user question -> retriever (top 3 chunks) -> prompt -> Llama 3.1 (Groq) -> answer

user sentence -> bag-of-words (CountVectorizer) -> MLP classifier -> service + confidence
                                                                   -> pre-fills the booking form
```

| Part | Technology |
|------|-----------|
| UI | Streamlit |
| RAG | LangChain, FAISS, Google Gemini embeddings, Groq (Llama 3.1 8B) |
| Intent classifier | scikit-learn `MLPClassifier`, `CountVectorizer` |
| Storage | SQLite |

## Project structure

```
app.py               Streamlit UI (3 tabs)
booking_logic.py     Booking rules, validation and SQLite storage
intent_classifier.py Neural-network intent classifier
rag.py               RAG chain (chunking, embeddings, retriever, LLM)
data/booking_docs.txt  Knowledge base for the chatbot
```

## Run it locally

```bash
git clone https://github.com/shahdamer067/smart-booking-system.git
cd smart-booking-system
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows: copy .env.example .env
# edit .env and add your keys
streamlit run app.py
```

Free API keys:
- Google AI Studio (Gemini embeddings): https://aistudio.google.com/apikey
- Groq (Llama 3.1): https://console.groq.com/keys

The booking and training tabs work without keys. Only the chatbot needs them.

## Notes and limitations

- The classifier is a small bag-of-words model trained on a starter set of examples (about 4 per service). It is meant to show how a trainable agent works, and it returns "not sure" when its confidence is low. Add more examples in the **Train Agent** tab to improve it.
- The chatbot only knows what is in `data/booking_docs.txt`.
- There is no user login. Anyone using the app can see and cancel all bookings.

## Ideas for next steps

- Let the chatbot create bookings through LLM tool calling.
- Replace bag-of-words with sentence embeddings for the classifier.
- Add tests for `booking_logic.py` and deploy to Streamlit Community Cloud.

## Author

Shahd Mohamed Amer, Computer Science (Intelligent Systems), Alexandria University.
