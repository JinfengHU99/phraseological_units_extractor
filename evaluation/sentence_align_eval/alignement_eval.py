import pandas as pd
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

data = []

with open("alignement_eval.txt","r") as f:
    for line in f:
        line = line.rstrip("\n").strip()
        if line:
            values = line.split("\t",1)
            if len(values) == 2:
                last_part = values[1].split("(",1)
                if len(last_part) > 1:
                    last_part = last_part[1].rstrip(")")
                else:
                    last_part = ""
                values.append(last_part)
            else:
                values.append("")
            data.append(values)
            
df = pd.DataFrame(data, columns = ["chinois","français","référence"])

regex_dict = {
    r".*l'alignement est.*" : "correct",
    r".*l'alignement n’est pas.*" : "incorrect"
}


df.iloc[:,2:] = df.iloc[:,2:].replace(regex_dict, regex = True)

df["prédiction"] = "correct"

df.to_csv("alignement_eval.csv", sep=",", encoding="utf-8")

#print(pd.crosstab(df["référence"], df["prédiction"]))
#print(precision_recall_fscore_support(df["référence"], df["prédiction"]))
print(accuracy_score(df["référence"], df["prédiction"]))
