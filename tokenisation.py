# coding: utf-8

import thulac
import spacy
from settings import SENTENCE_ALIGN_PATH,TOKENISATION_PATH

thu = thulac.thulac(seg_only=True)
nlp = spacy.load("fr_core_news_sm")

with open(SENTENCE_ALIGN_PATH, "r", encoding='utf-8') as input_file, \
        open(TOKENISATION_PATH, "w", encoding='utf-8') as output_file:
    chinese_text = ""
    french_text = ""

    for line in input_file:
        if "\t" in line:
            line = line.replace("\t", "|||")
            if chinese_text and french_text:
                chinese_tokens = thu.cut(chinese_text, text=True)
                french_doc = nlp(french_text)
                french_tokens = [token.text for token in french_doc]

                # Use space to separate each token
                chinese_tokens_str = ' '.join(chinese_tokens.split())
                french_tokens_str = ' '.join(french_tokens)

                output_line = f"{chinese_tokens_str}|||{french_tokens_str}\n"
                output_file.write(output_line)

            chinese_text, french_text = line.split("|||", 1)
        else:
            # If encountering a new line, append the current line to the previous French text
            french_text += line.strip()

    # Process the text at the end of the file
    if chinese_text and french_text:
        chinese_tokens = thu.cut(chinese_text, text=True)
        french_doc = nlp(french_text)
        french_tokens = [token.text for token in french_doc]

        # Use space to separate each token
        chinese_tokens_str = ' '.join(chinese_tokens.split())
        french_tokens_str = ' '.join(french_tokens)

        output_line = f"{chinese_tokens_str}|||{french_tokens_str}\n"
        output_file.write(output_line)
