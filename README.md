# MotiBot

## Motivational ChatBot Project

This project demonstrates a simple chatbot built using three different models, each with different architectures and techniques for natural language processing (NLP). The models are built using TensorFlow, PyTorch, and the Hugging Face Transformers library. They can be used to process user input and generate responses based on trained data.

## Project Structure

- **Bot_V1_experiments_tutorial**: A chatbot built with TensorFlow using an LSTM model for natural language understanding and generation.
- **Bot_V2**: Another version of the chatbot built with a different architecture (using Conv1D and LSTM layers) that uses a custom dataset from a training folder.
- **Bot_with_Transformer**: A chatbot utilizing GPT-2 (from the Hugging Face Transformers library) for generating responses using a pre-trained model.

## Prerequisites

To run this project, you will need the following dependencies:

- **Python 3.x**
- **TensorFlow 2.x**: For training and running the first two models.
- **PyTorch**: For running the GPT-2 model in Bot_with_Transformer.
- **Hugging Face Transformers**: For GPT-2.

You can install the required libraries using `pip`:

```bash
pip install tensorflow torch transformers nltk matplotlib
```

## Data

The first two models require a training dataset, which can be placed in a directory named `Training`. The chatbot will load the files from this directory and preprocess the data accordingly.

For **Bot_V1_experiments_tutorial**, the training data is loaded from a single file (`data.txt`). Ensure that this file is available in the working directory.

## How to Use

### Bot_V1_experiments_tutorial: TensorFlow LSTM Chatbot

1. Load the necessary data from the file `data.txt`.
2. Preprocess the text by tokenizing, lemmatizing, and removing stopwords and punctuation.
3. Train a model on the preprocessed data using TensorFlow.
4. The model generates responses to user input based on the trained data.

To run the model:

```bash
python code1.py
```

### Bot_V2: Custom Chatbot with TensorFlow

1. This model loads multiple text files from the `Training` folder, preprocesses them, and trains a custom chatbot using a combination of Conv1D and LSTM layers.
2. The trained model is saved as `Motibot2.keras`.

To run the model:

```bash
python code2.py
```

### Bot_with_Transformer: GPT-2 Chatbot

1. This version uses a pre-trained GPT-2 model from Hugging Face.
2. The model generates responses based on the input prompt.

To run the model:

```bash
python code3.py
```

## How It Works

- **Bot_V1_experiments_tutorial and Bot_V2**: Both of these models work by preprocessing text data, tokenizing it, and using an LSTM model to learn the relationship between input questions and expected answers. The model generates a response based on the input and the trained data.
- **Bot_with_Transformer**: This model uses the GPT-2 model, which is a transformer-based model. GPT-2 has been trained on a large corpus of text and can generate human-like responses to prompts. It generates responses word by word based on the input prompt.

## Training the Models

- **Bot_V1_experiments_tutorial and Bot_V2**: The training involves tokenizing the input text, creating sequences, padding them, and then feeding them into an LSTM or hybrid Conv1D/LSTM model. The models are trained using categorical cross-entropy loss and an Adam optimizer.

- **Bot_with_Transformer**: GPT-2 is a pre-trained model, so there is no training involved. The model generates responses using the existing weights.

## Model Evaluation and Visualization (Bot_V2)

For **Bot_V2**, you can visualize the accuracy and loss during training using the following function:

```python
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
```

This function plots a graph for the accuracy or loss of the model over the training epochs.

## Example Usage

To interact with the chatbot in **Bot_V1_experiments_tutorial** or **Bot_V2**, simply run the script and enter your queries. The bot will generate a response based on the model's training.

To interact with the GPT-2 chatbot in **Bot_with_Transformer**, you can type a prompt, and the model will generate a response based on that input.

```bash
Chatbot: Hi there! How can I help you?
You: Tell me a joke
Chatbot: Why did the chicken cross the road? To get to the other side.
```

## Future Improvements

- Add more data to the training set for better responses.
- Explore additional NLP models (like BERT or T5) for different use cases.
- Improve the chatbot's ability to handle multiple turns in a conversation.

## License

This project is open-source and available under the MIT License.
