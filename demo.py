import sentencepiece as spm
import torch
from fairseq.models.transformer import TransformerModel
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import gradio as gr

# Initialize the normalizer for Nepali text
remove_nuktas = False
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("ne", remove_nuktas=False)

# Load SentencePiece models
en2ne_sp = spm.SentencePieceProcessor()
en2ne_sp.load('en2ne/en2ne5000.model')

ne2en_sp = spm.SentencePieceProcessor()
ne2en_sp.load('ne2en/ne2en20000.model')

# Load Fairseq models
en2ne = TransformerModel.from_pretrained(
    'en2ne',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='en2ne/en2ne5000'
)
ne2en = TransformerModel.from_pretrained(
    'ne2en',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='ne2en/ne2en20000'
)

def en2ne_translate_sentence(sentence, reverse=False):
    tokenizer = en2ne_sp
    model = en2ne
    if reverse:
        sentence = normalizer.normalize(sentence)
        tokenizer = ne2en_sp
        model = ne2en

    # Tokenize the input sentence
    tokenized_sentence = tokenizer.encode(sentence, out_type=str)
    tokenized_sentence = ' '.join(tokenized_sentence)
    
    # Translate using the Fairseq model
    translation = model.translate(tokenized_sentence)

    # Decode the translation back to text
    translated_sentence = tokenizer.decode(translation.split())
    
    return translated_sentence

def translate(text, direction):
    if direction == "English to Nepali":
        return en2ne_translate_sentence(text, reverse=False)
    else:
        return en2ne_translate_sentence(text, reverse=True)

def clear_text_boxes(direction):
    return "", ""

with gr.Blocks() as demo:
    direction = gr.Dropdown(
        ["English to Nepali", "Nepali to English"], 
        label="Translation Direction", 
        value="English to Nepali", 
        interactive=True
    )
    with gr.Row():
        input_text = gr.Textbox(label="Input Text")
        output_text = gr.Textbox(label="Translated Text")

    translate_btn = gr.Button("Translate")
    translate_btn.click(translate, inputs=[input_text, direction], outputs=output_text)

    direction.change(clear_text_boxes, inputs=direction, outputs=[input_text, output_text])

demo.title = 'Neural Machine Translation (NMT) (English⇔Nepali)'
demo.description = demo.title
demo.launch()
