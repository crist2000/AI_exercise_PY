#use cmd to install the model: python -m spacy download en_core_web_sm
import spacy

#Named Entity Recognition(NER) example.
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K startup for $1 dollar")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)