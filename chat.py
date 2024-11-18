import gradio as gr
import ollama

def format_history(query, history, system_prompt):
    chat_history = [{"role" : "system", "content" : system_prompt}]
    for h in history:
        chat_history.append({"role" : "user", "content" : h[0]})
        chat_history.append({"role" : "assistant", "content" : h[1]})
    chat_history.append({"role" : "user", "content" : query})
    return chat_history

def generate_response(query, history, model, temperature, top_k, top_p):
    messages = format_history(query, history, "you are a helpful assistant")
    response = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    )
    message = ""
    for chunk in response:
        message += chunk['message']['content']
        yield message

# Model selection dropdown
model_dropdown = gr.Dropdown(
    choices=['qwen2.5:7b', 'llama3:8b', 'mistral:7b'],
    label="Select Model",
    value='qwen2.5:7b'
)

# Temperature slider
temperature_slider = gr.Slider(
    minimum=0.0,
    maximum=2.0,
    value=0.7,
    step=0.1,
    label="Temperature (Controls randomness: 0.0 = deterministic, 2.0 = most random)"
)

# Top-K slider
top_k_slider = gr.Slider(
    minimum=1,
    maximum=100,
    value=40,
    step=1,
    label="Top-K (Limits token selection to top K most probable tokens)"
)

# Top-P slider
top_p_slider = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=0.9,
    step=0.01,
    label="Top-P (Nucleus sampling - selects tokens whose cumulative probability exceeds P)"
)

# user interface
ui = gr.ChatInterface(
    fn=generate_response,
    title="Ollama Chat with Advanced Parameters",
    additional_inputs=[
        model_dropdown,
        temperature_slider,
        top_k_slider,
        top_p_slider
    ]
)

ui
