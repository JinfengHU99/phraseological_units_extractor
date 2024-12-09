from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import re
from settings import TOKENISATION_PATH,IDIOM_ALIGN_PATH

# Load model
model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

# Model parameters
align_layer = 8
threshold = 1e-3  # 调整阈值

# Output file
output_file = open(IDIOM_ALIGN_PATH, "w", encoding='utf-8')

# Dictionary to store mappings between Chinese idioms and French tokens
idiom_mapping = {}

# Read tokenisation.txt file
with open(TOKENISATION_PATH, "r", encoding='utf-8') as input_file:
    for line in input_file:
        # Split Chinese and French sentences using |||
        parts = line.strip().split("|||")

        # Check if both Chinese and French sentences are present
        if len(parts) != 2:
            # Skip lines with only Chinese without printing any message
            continue

        chinese_sentence, french_sentence = parts

        # Pre-process Chinese sentence
        sent_src = chinese_sentence.split()

        # Filter out words containing numbers or letters
        sent_src = [word for word in sent_src if not re.search(r'[0-9a-zA-Z]', word)]

        # Check if the sentence is not empty after filtering
        if not sent_src:
            continue

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
        sub2word_map_tgt = [i for i, word_list in enumerate(token_tgt) for x in word_list]

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
        align_words = set()
        for i, j in align_subwords:
            align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

        # Process the aligned words and update the idiom_mapping dictionary
        for s_i, t_i in sorted(align_words, key=lambda x: x[0]):
            chinese_token = sent_src[s_i]
            french_token = sent_tgt[t_i]

            # Check if the Chinese token is part of an idiom
            if len(chinese_token) >= 4:
                if chinese_token in idiom_mapping:
                    # Add the French token only if it's not already present
                    if french_token not in idiom_mapping[chinese_token]:
                        idiom_mapping[chinese_token].append(french_token)
                else:
                    idiom_mapping[chinese_token] = [french_token]

# Write the idiom_mapping to the output file
for chinese_idiom, french_tokens in idiom_mapping.items():
    french_translation = ' '.join(french_tokens)
    output_file.write(f"{chinese_idiom}\t{french_translation}\n")

# Close the output file
output_file.close()
