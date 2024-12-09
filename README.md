# Extraction of Chinese and French phraseological units

This project in NLP (Natural Language Processing) is implemented as part of the [DiCop](https://www.phraseologia.com/) project (Dictionary and Corpus of Phraseology) proposed by [**Mrs. LIU Lian**](https://lianchen.fr/projets.html).

In short, the goal of Mrs. Liu Lian's project is to create a multilingual dictionary. As part of this larger project, the current task involves extracting Chinese lexical units and determining their appropriate French translations through parallel text alignment.

To carry out the project's task, which is the extraction of Chinese and French phraseological units to be included in the DiCop project's digital dictionary, we used the corpus of the novel [**The Three-Body Problem (三体, Sān tǐ)**](https://en.wikipedia.org/wiki/The_Three-Body_Problem_(novel)) by the Chinese author [**LIU Cixin**](https://en.wikipedia.org/wiki/Liu_Cixin).

# Installation

Please see [requirements.txt](requirements.txt) for installation.

# Approach

In order to extract Chinese and French phraseological units, we first need two TXT files containing the corpus, one in [Chinese](corpus/corpus_zh) and the other in [French](corpus/corpus_fr).

If you want to directly extract corresponding phraseological units from the Chinese and French texts, simply run [run_scripts.py](run_scripts.py).

If you prefer to follow the execution logic step by step, please refer to the details below.

1.First, run [align_sentence.py](align_sentence.py). This step is aimed at achieving sentence alignment between the Chinese and French parallel texts, and the aligned sentence pairs will be generated in the [corpus](corpus) folder, specifically in the file [sentence_align.txt](corpus/sentence_align.txt). The primary method used is [**BertAlign**](https://github.com/bfsujason/bertalign?tab=readme-ov-file).

2.In the next step, execute [tokenisation.py](tokenisation.py), which includes the tokenization process for both French, using [**spaCy**](https://spacy.io/models/fr), and Chinese, using [**THULAC**](https://github.com/thunlp/THULAC-Python). The generated token alignment will be saved in the [corpus](corpus) folder in the file [tokenisation.txt](corpus/tokenisation.txt).

3.Finally, if you want to also examine the alignment of tokens, you can execute [align_token.py](align_token.py) and the file is [word_align.txt](corpus/word_align.txt) in the [corpus](corpus). Alternatively, to directly obtain the list of phraseological unit alignments from the corpus, run [align_idiom.py](align_idiom.py) in the same directory. The method used is [**awesome-align**](https://github.com/neulab/awesome-align) and its model [**aneuraz/awesome-align-with-co**](https://huggingface.co/aneuraz/awesome-align-with-co/tree/main).

# Performance 

See [idiom_eval.xlsx](evaluation/idiom_eval/idiom_eval.xlsx).

# Project Report

[Extraction des Expressions Idiomatiques Chinoises et Leurs Traductions Françaises Correspondantes dans le Cadre du Projet DiCoP](Extraction%20des%20Expressions%20Idiomatiques%20Chinoises%20et%20Leurs%20Traductions%20Françaises%20Correspondantes%20dans%20le%20Cadre%20du%20Projet%20DiCoP.pdf)
