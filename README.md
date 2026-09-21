# Multi-Task NLP model using LSTM


Datasets:

(i) Emotion Data: https://www.kaggle.com/datasets/nelgiriyewithana/emotions
(ii) Violence Data: https://www.kaggle.com/datasets/gauravduttakiit/gender-based-violence-tweet-classification?select=Train.csv
(iii) Hate Speech Data: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

# 🧠 Multi-Task Learning with NLP

> **A deep learning approach for jointly detecting emotions, violence, and hate/offensive speech from text.**

This project explores **Multi-Task Learning (MTL)** for Natural Language Processing by training a single neural network to perform three related text-classification tasks simultaneously:

* 😊 **Emotion Detection**
* ⚠️ **Violence Detection**
* 🚫 **Hate / Offensive Speech Detection**

Instead of building three completely independent models, the project uses **shared NLP representations** through a common embedding and LSTM backbone, followed by task-specific classification heads.

---

## 📌 Project Overview

Traditional NLP pipelines often train a separate model for every classification task. This project investigates a different approach: **learning shared linguistic representations across multiple related tasks**.

The architecture receives three task-specific inputs and processes them using shared layers:

```text
                    ┌─────────────────────┐
                    │   Input Text Data   │
                    └──────────┬──────────┘
                               │
                     Tokenization + Padding
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        Emotion Input    Violence Input     Hate Input
              │                │                │
              └────────────────┼────────────────┘
                               │
                    Shared Embedding
                         (128-D)
                               │
                       Shared LSTM
                          (64 units)
                               │
                 Global Average Pooling
                               │
                         Dropout (0.5)
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    Emotion Head         Violence Head         Hate Head
      6 classes             5 classes            3 classes
          │                    │                    │
       Softmax              Softmax              Softmax
```

The model is implemented using the **Keras Functional API** with multiple inputs and multiple outputs.

---

## 🎯 Tasks

### 1. Emotion Classification

The model predicts one of six emotions:

| Label | Emotion  |
| ----: | -------- |
|     0 | Sadness  |
|     1 | Joy      |
|     2 | Love     |
|     3 | Anger    |
|     4 | Fear     |
|     5 | Surprise |

The notebook works with a sampled dataset of **12,000 text samples** for this task.

### 2. Violence Classification

The model identifies five categories:

| Label | Category                     |
| ----: | ---------------------------- |
|     0 | Harmful Traditional Practice |
|     1 | Physical Violence            |
|     2 | Economic Violence            |
|     3 | Emotional Violence           |
|     4 | Sexual Violence              |

The original violence dataset contains substantially more samples, but the project constructs a balanced working subset of **12,000 samples**.

### 3. Hate / Offensive Speech Classification

The third task contains three categories:

| Label | Category         |
| ----: | ---------------- |
|     0 | Hate Speech      |
|     1 | Offensive Speech |
|     2 | Neither          |

A **12,000-sample** working dataset is used for training.

---

## 🏗️ Model Architecture

The project uses a shared representation-learning architecture.

### 1. Tokenization

Text from all three datasets is combined to build a shared vocabulary using Keras `Tokenizer`.

```python
tokenizer = Tokenizer()

tokenizer.fit_on_texts(
    pd.concat([
        emotion_df['text'],
        violence_df['text'],
        hate_df['text']
    ])
)
```

Each task's text is then converted into integer sequences.

### 2. Sequence Padding

All sequences are padded/truncated to a fixed length of **50 tokens**.

```python
max_length = 50

emotion_padded = pad_sequences(
    emotion_sequences,
    maxlen=max_length,
    padding='post'
)
```

The same procedure is applied to the violence and hate datasets.

### 3. Shared Embedding Layer

A common embedding layer converts token IDs into dense **128-dimensional word representations**.

```python
embedding_layer = keras.layers.Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim=128
)
```

The same embedding layer is reused across all three tasks.

### 4. Shared LSTM

The embedded sequences are passed through a shared LSTM with **64 hidden units**.

```python
shared_lstm = keras.layers.LSTM(
    64,
    return_sequences=True
)
```

Sharing this layer allows the model to learn linguistic patterns that can be useful across multiple classification tasks.

### 5. Global Average Pooling

The sequential LSTM representations are converted into fixed-length feature vectors using:

```python
GlobalAveragePooling1D()
```

### 6. Dropout

A dropout rate of **0.5** is applied before classification to reduce overfitting.

### 7. Task-Specific Output Heads

Three independent classification heads are used:

```python
emotion_output = Dense(
    6,
    activation='softmax',
    name='emotion_output'
)

violence_output = Dense(
    5,
    activation='softmax',
    name='violence_output'
)

hate_output = Dense(
    3,
    activation='softmax',
    name='hate_output'
)
```

Thus, the shared backbone learns common representations while each output head specializes in its respective task.

---

## ⚙️ Training Configuration

The model is trained using:

| Parameter           | Configuration                    |
| ------------------- | -------------------------------- |
| Framework           | TensorFlow / Keras               |
| Architecture        | Multi-Task LSTM                  |
| Embedding Dimension | 128                              |
| LSTM Units          | 64                               |
| Sequence Length     | 50                               |
| Dropout             | 0.5                              |
| Optimizer           | Adam                             |
| Loss                | Sparse Categorical Cross-Entropy |
| Epochs              | 10                               |
| Batch Size          | 4                                |
| Output Functions    | Softmax                          |

The three task losses are optimized simultaneously.

---

## 📊 Training Results

The recorded training run shows the following final **training-set accuracies after epoch 10**:

| Task                       | Training Accuracy |
| -------------------------- | ----------------: |
| 😊 Emotion                 |        **98.70%** |
| ⚠️ Violence                |        **99.92%** |
| 🚫 Hate / Offensive Speech |        **93.60%** |

The notebook records these values at the end of the 10th training epoch.

> **Note:** These are training accuracies from the recorded notebook run, not held-out test-set metrics. They should therefore not be interpreted as generalization performance.

The project also generates **normalized confusion matrices** for all three tasks to inspect class-level prediction behavior.

---

## 🔬 Data Processing Pipeline

The overall pipeline is:

```text
Raw Text
   │
   ▼
Data Loading
   │
   ▼
Column Standardization
   │
   ▼
Null-Value Checking
   │
   ▼
Dataset Sampling
   │
   ▼
Text Preprocessing
   │
   ▼
Shared Tokenizer
   │
   ▼
Integer Sequences
   │
   ▼
Padding / Truncation
   │
   ▼
Shared Embedding
   │
   ▼
Shared LSTM
   │
   ▼
Global Average Pooling
   │
   ▼
Dropout
   │
   ├───────────────┬────────────────┐
   ▼               ▼                ▼
Emotion         Violence           Hate
Classifier      Classifier       Classifier
```

---

## 🧩 Why Multi-Task Learning?

Multi-task learning allows related tasks to share information during training.

For example, understanding linguistic patterns associated with emotional language may also provide useful representations for detecting harmful or offensive language.

Instead of learning:

```text
Task A → Model A
Task B → Model B
Task C → Model C
```

this project explores:

```text
                 Shared Representation
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         Emotion      Violence       Hate
```

This provides a practical demonstration of how **parameter sharing** can be used for multiple NLP objectives.
---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ZORAKEN/nlp.git
cd nlp
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nltk jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Multi_Task_Learning_with_NLP.ipynb
```

### 4. Run the notebook

Run the cells sequentially to reproduce the preprocessing, model construction, training, predictions, and evaluation.

> **Important:** The notebook currently references datasets using Colab-style paths such as `/content/...`. You may need to update the dataset paths when running it locally.

---

## 📈 Evaluation

The project evaluates predictions using:

* Task-specific classification accuracy
* Confusion matrices
* Class-level prediction analysis

The model generates predictions independently for each output:

```python
prediction = model.predict({
    'emotion_input': emotion_input,
    'violence_input': violence_input,
    'hate_input': hate_input
})
```

The predicted class is then obtained using `argmax` for each task.

---

## 💡 Key Concepts Demonstrated

This project provides hands-on implementation of several important NLP and deep-learning concepts:

* **Multi-Task Learning**
* **Shared Representation Learning**
* **Word Embeddings**
* **LSTM Networks**
* **Sequence Modeling**
* **Text Tokenization**
* **Sequence Padding**
* **Multi-Input / Multi-Output Neural Networks**
* **Softmax Classification**
* **Sparse Categorical Cross-Entropy**
* **Confusion Matrix Analysis**
* **Parameter Sharing**
---

## 📚 Notebook

**Main implementation:**

[Multi_Task_Learning_with_NLP.ipynb](https://github.com/ZORAKEN/nlp/blob/main/Multi_Task_Learning_with_NLP.ipynb)

---

## ⭐ Project Highlights

> **A multi-task NLP system that learns shared linguistic representations and simultaneously performs emotion, violence, and hate/offensive speech classification using a shared Embedding + LSTM architecture.**



Multi-Task model generates output for each task individually
(iii) Shared Core Layers:

Multi-Task models functionality is that a sinlge model can do multiple related tasks. To exhibit this functionality the core layers of the model like (Embedding, LSTM, pooling, dropout etc) are shared among all the tasks and are not seperately available for them
