# coding: utf-8

from bertalign import Bertalign
from settings import CORPUS_ZH_PATH, CORPUS_FR_PATH, SENTENCE_ALIGN_PATH


def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def clean_text(text):
    if isinstance(text, list):
        return [t.strip() for t in text]
    elif isinstance(text, str):
        return text.strip()
    else:
        return text


sentence_zh = read_text_from_file(CORPUS_ZH_PATH)
sentence_zh = clean_text(sentence_zh)
sentence_fr = read_text_from_file(CORPUS_FR_PATH)
sentence_fr = clean_text(sentence_fr)

aligner = Bertalign(sentence_zh, sentence_fr)
aligner.align_sents()
#aligner.print_sents()

with open(SENTENCE_ALIGN_PATH, "w", encoding="utf-8") as filealign:
    for align in aligner.result:
        src_line = aligner._get_line(align[0], aligner.src_sents)
        tgt_line = aligner._get_line(align[1], aligner.tgt_sents)
        filealign.write(f"{src_line}\t{tgt_line}\n")

