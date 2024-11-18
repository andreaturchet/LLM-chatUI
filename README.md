# Ollama Chat with Advanced Parameters

This project creates a flexible chatbot interface using **Gradio** and **Ollama's API**, allowing users to interact with advanced machine learning models. Users can configure multiple parameters, such as model selection, temperature, top-k, and top-p, to customize the behavior of the assistant.

---

## Features

1. **Dynamic Chat Interface**:
   - Built with `gr.ChatInterface`, which tracks conversation history and provides a seamless user experience.

2. **Model Selection**:
   - Choose from multiple pre-trained models like `qwen2.5:7b`, `llama3:8b`, and `mistral:7b` using a dropdown menu.

3. **Response Configuration**:
   - Adjust response behavior using advanced sampling techniques:
     - **Temperature**: Controls randomness in the model's output.
     - **Top-K**: Limits token selection to the top K most probable options.
     - **Top-P**: Enables nucleus sampling to focus on the most probable responses.

4. **Streaming Responses**:
   - Messages are streamed in real-time for an engaging chat experience.


---

## Author:
Andrea Turchet
