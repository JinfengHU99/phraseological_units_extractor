from transformers import AutoModel, AutoTokenizer
import itertools
import torch
from settings import TOKENISATION_PATH,WORD_ALIGN_PATH

# Load model
model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

# Model parameters
align_layer = 8
threshold = 1e-3

# Output file
output_file = open(WORD_ALIGN_PATH, "w", encoding='utf-8')

# Read tokenisation.txt file
with open(TOKENISATION_PATH, "r", encoding='utf-8') as input_file:
    for line in input_file:
        # Split Chinese and French sentences using |||
        parts = line.strip().split("|||")

        # Check if both Chinese and French sentences are present
        if len(parts) != 2:
            continue

        chinese_sentence, french_sentence = parts

        # Pre-process Chinese sentence
        sent_src = chinese_sentence.split()
        token_src = [tokenizer.tokenize(word) for word in sent_src]
        wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
        ids_src = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                              model_max_length=tokenizer.model_max_length, truncation=True)['input_ids']
        sub2word_map_src = [i for i, word_list in enumerate(token_src) for x in word_list]

        # Pre-process French sentence
        sent_tgt = french_sentence.split()
        token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]
        wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True,
                                              model_max_length=tokenizer.model_max_length)['input_ids']

        # Map each French token to corresponding Chinese tokens
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt.extend([i] * len(word_list))

        # Alignment
        model.eval()
        with torch.no_grad():
            out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = dict()
        for i, j in align_subwords:
            if sub2word_map_tgt[j] not in align_words:
                align_words[sub2word_map_tgt[j]] = sub2word_map_src[i]

        # Debug prints
#         print("Chinese sentence:", chinese_sentence)
#         print("Chinese tokens:", sent_src)
#         print("French sentence:", french_sentence)
#         print("French tokens:", sent_tgt)
#         print("Aligned words:", {sent_tgt[t_i]: sent_src[s_i] for t_i, s_i in align_words.items()})

        # Write the alignment to file
        for t_i, s_i in align_words.items():
            # Check if the index is within the range of French sentence
            if 0 <= t_i < len(sent_tgt):
                output_file.write(f"{sent_tgt[t_i]}\t{sent_src[s_i]}\n")

# Close the output file
output_file.close()
