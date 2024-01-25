import json

path1 = "./raw_output/refcoco+_val.json"
path2 ="./clean_output/refcoco+_val.json"

with open(path1,'r') as file:
    content = json.load(file)
outputs = []
ref_ids = []
predictions = []
out = {}
# content = content['predictions']
# print(content)
for i in range(len(content)):
    data = {}
    output = content[i]['text_output']
    ref_id = content[i]['ref_id']
    for j in range(len(output)):
        if output[j:j+7] == ' It is ' :
            output = output[0:j] + output[j+7:]
        if output[j:j+6] == 'It is ' :
            output = output[0:j] + output[j+6:]
        if output[j:j+6] == 'it is ' :
            output = output[0:j] + output[j+6:]
        if output[j:j+9] == 'There is ':
            output = output[0:j] + output[j+9:]
        if output[j:j+2] == '. ':
            output = output[0:j] 
        if output[j:j+1] == '.':
            output = output[0:j] 
        if output[j:j+2] == ', ':
            output = output[0:j] 
        if output[j:j+11] == 'at region0 ':
            output = output[0:j] +  output[j+11:]
        if output[j:j+8] == 'istant: ':
            output = output[:j] +  output[j+8:]
        if output[j:j+3] == 'is ':
            output = output[j+3:] 
            break
    #     #delete it
    data["sent"] = output
    data["ref_id"] = ref_id
    predictions.append(data)
out["predictions"] = predictions
with open(path2,'w',encoding='ascii') as f:
    json.dump(out,f)