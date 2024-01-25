import json
from score import score

path1 = "./raw_output/refcocog_test.json"
path2 ="./modify_output/refcocog_test.json"
with open(path1,'r') as file:
    content = json.load(file)

gt = {}
gts = []

for i in range(len(content)):
    id = str(i)
    out = []
    answer = ''
    for j in range(len(content[i]["gt"])):
        if j > 0:
            answer = answer + ' And ' + content[i]["gt"][j]  
        else:
            answer = content[i]["gt"][j]
    out.append(answer)
    gts.append(answer)
    gt[id] = out


with open(path2,'r') as file:
    content = json.load(file)

output = {}
outputs = []
for i in range(len(content["predictions"])):
    id = str(i)
    out = []
    out.append(content["predictions"][i]["sent"])
    outputs.append(content["predictions"][i]["sent"])
    output[id] = out

(P, R, F), hashname = score(gts, outputs, lang="en", return_hash=True)
print(
f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")