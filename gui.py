import gradio as gr
from main import generate_article


def gradio_interface(urls, query):
    urls = urls.split("\n")
    response = generate_article(urls, query)
    return response


def run_gradio():
    css = """
    h1 {
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Генератор статей")
        with gr.Row():
            url_input = gr.Textbox(
                label="URL-адреса",
                placeholder="Введите URL-адреса, каждый адрес с новой строки",
                lines=4,
            )
            query_input = gr.Textbox(
                label="Тема статьи", placeholder="Введите тему статьи"
            )
        output = gr.Textbox(label="Сгенерированная статья", lines=20)
        generate_button = gr.Button("Сгенерировать статью")
        generate_button.click(
            gradio_interface, inputs=[url_input, query_input], outputs=output
        )

    demo.launch(
        server_name="0.0.0.0",
        share=True,
    )


if __name__ == "__main__":
    run_gradio()
