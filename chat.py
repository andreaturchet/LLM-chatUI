import gradio as gr
import ollama

def format_history(query, history, system_prompt, max_history=5):
    """
    Format chat history and limit its length.
    """
    chat_history = [{"role": "system", "content": system_prompt}]
    for h in history[-max_history:]:  # Limit to the last max_history exchanges
        chat_history.append({"role": "user", "content": h[0]})
        chat_history.append({"role": "assistant", "content": h[1]})
    chat_history.append({"role": "user", "content": query})
    return chat_history

def generate_response(query, history, model, temperature, top_k, top_p):
    """
    Generate a response from the model and update the conversation history.
    """
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

    partial_message = ""
    for chunk in response:
        partial_message += chunk['message']['content']
        # Update the latest message in history and yield
        updated_history = history + [(query, partial_message)]
        yield updated_history

# Create Gradio Blocks Interface
with gr.Blocks() as ui:
    gr.Markdown("## AI Chatbot with Adjustable Parameters")

    # Chatbot interface
    chatbot = gr.Chatbot(height=700)
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            label="Your Message"
        )
        clear = gr.Button("Clear Chat")

    # Settings for the chatbot
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=['qwen2.5:7b', 'qwen2.5-coder:14b'],
                label="Select Model",
                value='qwen2.5-coder:14b'
            )
        with gr.Row():
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=40,
                step=1,
                label="Top-K"
            )
            top_p_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.01,
                label="Top-P"
            )

    # Initialize conversation history
    state = gr.State([])

    # Define interaction logic
    def respond(message, history, model, temperature, top_k, top_p):
        """
        Handle user input and generate a response.
        """
        response_stream = generate_response(
            query=message,
            history=history,
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        for updated_history in response_stream:
            yield "", updated_history, updated_history

    # Link user input and responses
    msg.submit(
        respond,
        inputs=[msg, state, model_dropdown, temperature_slider, top_k_slider, top_p_slider],
        outputs=[msg, chatbot, state] 
    )

    # Clear chat button
    clear.click(lambda: ([], []), inputs=[], outputs=[chatbot, state])

# Launch the application
ui.launch()

