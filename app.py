import os
import gradio as gr
from transformer import PyTorchTransformer

MODEL_PATH = "model.pth"
transformer = PyTorchTransformer()


def state():
    if transformer.is_trained:
        return f"Modelo cargado | vocab={transformer.vocab_size} | device={transformer.device}"
    return "Sin modelo entrenado"


def generate(seed, max_chars, temperature, top_k):
    if not transformer.is_trained:
        return "Primero entrena el modelo"
    return transformer.generate(seed, int(max_chars), float(temperature), int(top_k))


def train_model(corpus, block_size, d_model, n_head, n_layer, d_ff, max_steps, lr):
    try:
        from tokenizers import TokenizerFactory, TokenizerConfig, TokenizerType

        transformer.__init__(
            block_size=int(block_size),
            d_model=int(d_model),
            n_head=int(n_head),
            n_layer=int(n_layer),
            d_ff=int(d_ff),
            max_steps=int(max_steps),
            learning_rate=float(lr),
        )

        tokenizer = TokenizerFactory.create(TokenizerConfig(
            vocab_size=4096,
            add_special_tokens=True,
        ))
        transformer.tokenizer = tokenizer

        lines = [l.strip() for l in corpus.split("\n") if l.strip()]
        for line in lines:
            transformer.train(line)

        tokenizer.train(lines, 4096)
        transformer.fit()

        if transformer.is_trained:
            transformer.save(MODEL_PATH)
            return f"Entrenado | vocab={transformer.vocab_size} | steps={max_steps}"
        return "Entrenamiento falló"
    except Exception as e:
        return f"Error: {e}"


def save_model():
    if transformer.is_trained:
        transformer.save(MODEL_PATH)
        return "Modelo guardado"
    return "Nada que guardar"


def load_model():
    if os.path.exists(MODEL_PATH):
        transformer.load(MODEL_PATH)
        return f"Modelo cargado | vocab={transformer.vocab_size}"
    return "No hay modelo guardado"


with gr.Blocks(title="Rubidium API") as demo:
    gr.Markdown("# Rubidium API - Transformer mini-GPT")

    with gr.Tab("Chat"):
        gr.Markdown(state())
        seed = gr.Textbox(label="Prompt", value="Hola")
        max_chars = gr.Slider(10, 500, value=200, label="Max chars")
        temperature = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
        top_k = gr.Slider(1, 100, value=20, label="Top K")
        btn_gen = gr.Button("Generar")
        output = gr.Textbox(label="Resultado")
        btn_gen.click(generate, inputs=[seed, max_chars, temperature, top_k], outputs=output)

    with gr.Tab("Entrenar"):
        corpus = gr.Textbox(label="Corpus", lines=10, value="Hola mundo")
        with gr.Row():
            block_size = gr.Number(label="Block Size", value=128)
            d_model = gr.Number(label="D Model", value=512)
            n_head = gr.Number(label="N Head", value=8)
        with gr.Row():
            n_layer = gr.Number(label="N Layer", value=8)
            d_ff = gr.Number(label="D FF", value=2048)
        with gr.Row():
            max_steps = gr.Number(label="Max Steps", value=6000)
            lr = gr.Number(label="Learning Rate", value=0.0001)
        btn_train = gr.Button("Entrenar")
        train_output = gr.Textbox(label="Estado")
        btn_train.click(train_model,
                       inputs=[corpus, block_size, d_model, n_head, n_layer, d_ff, max_steps, lr],
                       outputs=train_output)

    with gr.Tab("Modelo"):
        btn_save = gr.Button("Guardar")
        btn_load = gr.Button("Cargar")
        model_status = gr.Textbox(label="Estado")
        btn_save.click(save_model, outputs=model_status)
        btn_load.click(load_model, outputs=model_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
