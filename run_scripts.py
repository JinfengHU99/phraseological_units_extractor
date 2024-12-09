import subprocess
import os
from settings import SENTENCE_ALIGN_PATH,TOKENISATION_PATH,IDIOM_ALIGN_PATH

scripts_to_run = [
    "align_sentence.py",
    "tokenisation.py",
    "align_idiom.py",
]

output_files = [
    SENTENCE_ALIGN_PATH,
    TOKENISATION_PATH,
    IDIOM_ALIGN_PATH
]

for script, output_file in zip(scripts_to_run, output_files):
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        if os.path.exists(output_file):
            print(f"Success! File generated: {output_file}\n")
        else:
            print(f"Warning: {script} ran successfully, but {output_file} was not found.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break