# TrollGuard: Frontend vs Backend – A Beginner's Guide

## What is the Frontend?

The **frontend** is the part of an application that users see and interact with. It includes:

- Buttons, text boxes, tables, tabs
- Layout, colours, styling
- All user input (typing, clicking, uploading files)

The user "talks" to the application through the frontend.

---

## What is the Backend?

The **backend** is the logic that runs behind the scenes on a server. It:

- Loads and saves data
- Runs the model (predictions)
- Trains the model
- Handles file processing (CSV, chat files)
- Manages storage (database or files)

The backend processes the data and sends results back to the frontend to display.

---

## How TrollGuard Handles Frontend and Backend

TrollGuard uses **Streamlit**, which lets you build both frontend and backend in the **same Python file**. The distinction is still clear:

### Frontend (What the User Sees and Clicks)

These Streamlit components in `app.py` are the **frontend**:

| Component | Purpose |
|-----------|---------|
| `st.title`, `st.caption` | Page header and subtitle |
| `st.tabs` | Tabs (Predict, Chat analysis, Train model, About) |
| `st.text_area` | Text input boxes |
| `st.file_uploader` | File upload (CSV, TXT) |
| `st.button` | Buttons (Train, Analyze chat) |
| `st.dataframe` | Tables showing results |
| `st.download_button` | Download results as CSV |
| `st.sidebar` | Sidebar with model status and stats |
| `st.expander` | Collapsible sections |
| `st.progress` | Progress bar for batch processing |

---

### Backend (The Logic and Processing)

The **backend** is the Python logic that runs when the user interacts:

| Component | Location | Purpose |
|-----------|----------|---------|
| `load_parsed_datasets()` | `core/data_loader.py` | Load datasets |
| `clean_text()` | `core/model_utils.py` | Preprocess text |
| `load_model_and_vectoriser()` | `core/model_utils.py` | Load TF-IDF and model |
| `predict_text()`, `predict_batch()` | `core/model_utils.py` | Run predictions |
| `parse_chat_from_string()` | `core/chat_parser.py` | Parse chat formats |
| `TfidfVectorizer`, `LogisticRegression` | `app.py` / `train_model.py` | Train the model |
| `joblib.dump()` | `app.py` | Save the model |

---

## How They Work Together

```
User types/pastes text  →  Frontend (st.text_area)
                              ↓
User clicks or submits  →  Streamlit reruns the Python script
                              ↓
Backend (predict_text, predict_batch)  →  Model returns 0 or 1
                              ↓
Frontend (st.success, st.dataframe)  →  User sees the result
```

1. **Frontend** collects user input.
2. **Streamlit** reruns the script when the user acts (e.g. clicks).
3. **Backend** runs the model and produces results.
4. **Frontend** displays those results.

---

## Summary for Students

| Term | In TrollGuard |
|------|---------------|
| **Frontend** | Streamlit UI: `st.title`, `st.text_area`, `st.button`, `st.dataframe`, etc. |
| **Backend** | Python logic in `app.py` and `core/`: data loading, model, predictions |

In TrollGuard, both are in the same Python codebase. Streamlit creates the web interface and runs your backend logic when users interact with it.
