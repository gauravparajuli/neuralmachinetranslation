import sentencepiece as spm
import torch
from fairseq.models.transformer import TransformerModel
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# we normalize nepali text before passing it to the model
remove_nuktas=False
factory=IndicNormalizerFactory()
normalizer=factory.get_normalizer("ne",remove_nuktas=False)

# Load SentencePiece model
en2ne_sp = spm.SentencePieceProcessor()
en2ne_sp.load('en2ne/en2ne5000.model')

ne2en_sp = spm.SentencePieceProcessor()
ne2en_sp.load('ne2en/ne2en20000.model')

# Load Fairseq model
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

print(type(ne2en))

def en2ne_translate_sentence(sentence, reverse=False):
    tokenizer = en2ne_sp
    model = en2ne
    if reverse==True:
        sentence = normalizer.normalize(sentence)
        tokenizer = ne2en_sp
        model = ne2en

    # Tokenize the input sentence
    tokenized_sentence = tokenizer.encode(sentence, out_type=str)
    tokenized_sentence = ' '.join(tokenized_sentence)
    
    # Translate using the Fairseq model
    translation = model.translate(tokenized_sentence)

    print(translation)
    
    # Decode the translation back to text
    translated_sentence = tokenizer.decode(translation.split())
    
    return translated_sentence

# Example usage
input_sentence = "Hello, how are you?"
input_sentence2 = "मलाई जाँडो भयो।"
# translated_sentence = en2ne_translate_sentence(input_sentence)
translated_sentence = en2ne_translate_sentence(input_sentence2, reverse=True)
print(f"Translated sentence: {translated_sentence}")