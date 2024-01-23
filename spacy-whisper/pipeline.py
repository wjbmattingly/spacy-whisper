import spacy
from spacy.tokens import Doc
from spacy.tokens import Token

Token.set_extension("my_metadata", default=None)

class Whisper:
    def __init__(self, lang, model=None)
        if model:
            self.nlp = spacy.load(model)
        else:
            self.nlp = spacy.blank("en")
    def create_doc(self, whisper_output):
        self.text = whisper_output["text"]
        doc.has_unknown_spaces = True
        for i, entry in enumerate(entries):
            doc[i]._.my_metadata = entry["some_index"]
        doc = Doc(nlp.vocab, words=[entry["text"] for entry in entries])