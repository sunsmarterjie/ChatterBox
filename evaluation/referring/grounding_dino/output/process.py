import json

path1 = "./raw_output/refcocog_test.json"
path2 ="./modify_output/refcocog_test.json"

with open(path1,'r') as file:
    content = json.load(file)
outputs = []
ref_ids = []
predictions = []
out = {}
for i in range(len(content)):
    data = {}
    output = content[i]['text_output']
    ref_id = content[i]['ref_id']
    data["sent"] = output
    data["ref_id"] = ref_id
    predictions.append(data)
out["predictions"] = predictions
with open(path2,'w',encoding='ascii') as f:
    json.dump(out,f)