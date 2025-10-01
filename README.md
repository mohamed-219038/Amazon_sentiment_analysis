# Sentiment Analysis (RNN & LSTM) on Amazon Fine Food Reviews

This project predicts customer sentiment (1–5 stars) from the **Amazon Fine Food Reviews** dataset using **deep learning models (RNN & LSTM)**.

---

## 🔹 Steps
1. Preprocess text (clean, tokenize, pad)
2. Train **RNN** and **LSTM** models
3. Evaluate with accuracy & confusion matrix
4. Run demo predictions for custom sentences

---

# Sentiment Analysis with RNN & LSTM on Amazon Reviews

This project uses the **Amazon Fine Food Reviews Dataset** to classify customer reviews into **1–5 star ratings** using two deep learning models:
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**

---

## 🚀 Steps

1. **Data Preprocessing**
   - Load dataset (Text + Score columns)
   - Clean text (lowercase, remove punctuation, stopwords)
   - Tokenize and pad sequences
   - Convert labels (1–5 stars → one-hot encoded)

2. **Model Building**
   - RNN model with Embedding + SimpleRNN + Dense
   - LSTM model with Embedding + LSTM + Dense
   - Add Dropout to reduce overfitting

3. **Training**
   - Train both models on the preprocessed dataset
   - Monitor training/validation accuracy and loss

4. **Evaluation**
   - Accuracy score
   - Classification report
   - Confusion matrix (true vs predicted labels)

5. **Demo**
   - Enter any custom review sentence
   - Both RNN & LSTM models predict sentiment (1–5 stars)

---

## 🧩 Example Prediction
```python
predict_sentence(
    "The food was absolutely wonderful, I loved it!",
    tokenizer, rnn_model, maxlen
)

# Output
Sentence: Good Quality and very Delight and Great product and recommended!
Predicted Sentiment (RNN): 4 Stars
Predicted Sentiment (LSTM): 4 Stars
