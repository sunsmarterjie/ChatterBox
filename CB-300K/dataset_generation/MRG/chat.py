import os


def get_start_idx(dir):
    filenames=os.listdir(dir)
    print(filenames)
    filenames.sort(key=lambda x: int(x[:-4]))
    print(filenames)
    last_file=filenames[-1].split('.')[0]
    start_id=int(last_file)+1
    return start_id


import datetime
import json
import os
import openai

# openai.api_base='https://ngapi.xyz/v1'
# openai.api_key='sk-07kiWkUHpgcUzub0D2A776B75dB540AbAf166e54A11f7eF7'
openai.api_base = 'https://api.ngapi.top/v1'#v1
openai.api_key='sk-rUCYa5a7mdd2DE8A6fBe7337Ec0945D29b3f785716807eDd' #'sk-tedEG7qsdmOJfz7K2d6300B4842e43Df814eC21c6a8e4411' #'sk-Lb72otrreGCr6Dp76a68F7F236F54008A01bAf11B148E162'#'sk-07kiWkUHpgcUzub0D2A776B75dB540AbAf166e54A11f7eF7'

def supplement_1():
    start = 'Here are some cases and requirements you must pay attention to.\n'
    question_bbox = '1. If you think the bounding box or some expressions given in the question directly reveals the answer, do not present this information in the question. Here are some cases, while more examples are not given.####'
    question_bbox1 = '(1)	If the answer is about the attribute (such as color, shape, motion, etc) of an object, resent bounding boxes of all objects. For example: \n ' \
                     '"""correct example: (ask the color of the object, present all object) Question1: What is the color of the bird on the tree? <bird: [100,200,250,300], tree: [50,47,320,480]>Answer1: The color of the bird on the tree is brown. <bird: [100,200,250,300]> \n' \
                     'incorrect example: (the coordinates of the tree mentioned in the question is not given) Question1: What is the color of the bird on the tree? <tree: [50,47,320,480] > Answer1: The color of the bird on the tree is brown. <bird: [100,200,250,300]>"""\n'
    question_bbox2 = '(2)	If the question aims to locate an object, do not give the coordinates of this object, unless there are some objects that can’t be distinguished without coordinates.For example: \n' \
                     '"""correct example: (not present the bounding box of the object as the answer) Question1: Is the white cup on the right of the man? <man :[99,40,201,220]> Answer1: No, the white cup is on the left of the man. <cup: [25,100,51,124]> \n' \
                     'incorrect example: (present the bounding box of the object as the answer) Question1: Is the white cup on the right of the man? <man :[99,40,201,220], white cup: [25,100,51,124]> Answer1: No, the white cup is on the left of the man. <white cup: [25,100,51,124]>"""\n'
    question_bbox3 = '(3)	If the detailed description in the question directly reveals the answer (so you do not have to search for answer in the relationships information), do not present this information in the question. For example: \n' \
                     '"""correct example: (not present the bounding box of the object as the answer) Question1: Where is the white car? <white car: [99,100,300,220]> Answer1: The white car is parking on the street. <street: [71,194,410,301]> \n' \
                     'incorrect example: (not present the bounding box of the object as the answer) Question1: Where is the white car parking on the street? <car: [99,100,300,220], street: [71,194,410,301]> Answer1: The car is parking on the street. <street: [71,194,410,301]>"""\n'
    answer_bbox = '2. You must only present the coordinates of object to the question, do not show other objects’ coordinates. For example:\n' \
                  '"""correct example: (only give the coordinates of the bird) Question1: What is the color of the bird on the tree? <bird: [100,200,250,300], tree: [50,47,320,480]> Answer1: The color of the bird on the tree is brown. <bird: [100,200,250,300]> \n' \
                  'incorrect example: (give the coordinates of bird and tree) Question1: What is the color of the bird on the tree? <bird: [100,200,250,300], tree: [50,47,320,480]> Answer1: The color of the bird on the tree is brown. <bird: [100,200,250,300], tree: [50,47,320,480]>"""\n'
    all_obj = '3. If some objects can be the answer to the question, list them all in the answer. Attention, these objects may occur in different relationships. For example:\n' \
              'given information: pedal[337, 388, 358, 409] on black bicycle[168, 211, 479, 483] ; wheel[365, 317, 479, 484] on black bicycle[168, 211, 479, 483]; spokes[371, 325, 469, 474] on wheel[365, 317, 479, 484]\n' \
              '"""correct example: Question5: What is the object on the black bicycle? <black bicycle:[168, 211, 479, 483]>Answer5: The objects on the black bicycle are a chain, a pedal, a wheel with spokes, and a smiling and hanging out girl is leaning on it. <pedal:[337, 388, 358, 409], wheel:[365, 317, 479, 484], spokes:[371, 325, 469, 474] >. Reason: list all objects on black bicycle with a specific position, especially include spokes, because the wheel[365, 317, 479, 484] spokes on is on black bicycle[168, 211, 479, 483], which means spokes is on bicycle."""\n' \
              '"""incorrect example: Reason: only list a part of objects on the black bicycle, not only miss the objects that has direct relationship (wheel), but miss the objects that has indirect relationship (spokes) Question5: What is the object on the black bicycle? <black bicycle:[168, 211, 479, 483]>. Answer5: The objects on the black bicycle are a chain, a pedal, a wheel with spokes, and a smiling and hanging out girl is leaning on it. <pedal:[337, 388, 358, 409], wheel:[365, 317, 479, 484], spokes:[371, 325, 469, 474] >"""'
    complex_inference = '4. Try to avoid complex inference about the relationships, if you are not certain about the answer, do not ask this question.\n'
    no_bbox = '5. If the answer is not corresponding to object in the image, do not output any bounding box.\n'
    dont_know = "6. If the answer is 'the information is not given', replace the expression with 'I do not know'.\n"
    rel = '7. Be careful of the relationship between object, especially when the statement is different from the given relationships. For example:\n' \
          'given statement: tan dirt[0, 86, 500, 498] on feet[189, 457, 304, 492] \n' \
          'correct example: (understand the position relationship between object, in this case, feet is under dirt).Question10: What is the object on the tan dirt? <tan dirt:[0, 86, 500, 498]>. Answer10: I don’t know about it since the information is not given. But I can tell feet is under dirt.\n' \
          'incorrect example: (wrong relationship between objects, just copy the given statement). Question10: What is the object on the tan dirt? <tan dirt:[0, 86, 500, 498]>. Answer10: The object on the tan dirt is feet. <feet:[189, 457, 304, 492]>\n'
    bbox = '8.DO NOT MAKE ANY INFERENCE BASED ON COORDINATES!!! Do not consider two objects as the same thing, even if they have similar category names and bounding boxes, just consider them as two seperate stuffs.'
    return start + question_bbox + question_bbox1 + question_bbox2 + question_bbox3 + answer_bbox + all_obj + complex_inference + no_bbox + dont_know + rel + bbox



def task1(prompt):
    st2 = 'You are an AI visual assistant, and you are seeing a single image. You know several descriptions of the image, each describing a specific relationship in the image you are observing. The relationship is written as "subobject relation object", subobject and object are composed of name, bounding box. There may also be some attributes to subobject and object, such as "black cat [99,78,167,187] on desk [0,60,199,230]".' \
          'Please note that there may be several objects of the same category in an image, and we use "name_number" to present these objects, such as "man_1","man_2" to describe different men. Attention, attributes of an object may just occur in some relationships. For example: man_1[121,122,234,345] and running man_1[121,122,234,345] are the same object, because they share same category name and bounding box.' \
          'Specific information is as follow:\n'
    relationship_def = ' Relationship: the relationship between two objects, which written as "sub_object relation object". \n'
    st3 = 'Your task is to modify the relationships. Most of the relationships are correct, while there may be some error in these relationship information, what you need to do is to correct the incorrect ones based on common sense. ' \
          'There are only two kinds of mistakes: (1)The relationship is between same objects(similar names and bounding boxes), which means this relationship may not exist. (2)The relationship may be inverse between these two objects. Judge whether the relationship is inverse by the name of object. Wrong cases are not thoroughly listed. ' \
          'Please output (1)how to correct(that can be modify or delete) each incorrect relationship;(2)"@@@"(3)the relationship information after process.' \
          ' Answers from the above requirements should be outputed in order.The format of (3) is strictly defined: add "$$" between (1)subobject and relationship;(2)relationship and object. For example:man wears blue pants should be written as man $$ wears $$ blue pants. ' \
          'Form the output in the format of python list, for example:[man on tshirt, boy plays basketball].\n'
    st4 = 'Only modify relationships that have clear error that belongs to the mistake type as I mentioned above, do not change those you are not sure whether they are correct or how to modify.' \
          ' Also, do not make unnecessary changes, especially for the right ones. For example, do not change "man has foot" to "foot has man". For example,"a man is on a hat" is obviously wrong, but "cup is on the right of the girl"may be correct, so you should not change the later. ' \
          'Remain all useful information, do not delete any attribute of objects. ' \
          'After modifying, compare result and input and consider their retionality, if some output is worse than input, use input as output.\n'
    task=st2+relationship_def+str(prompt)+st3+st4
    return task

def task2(times):
    st3 = 'What you need to do is ask '
    st4 = ' new questions in order with the modified relationships generated in former task, each based on the answer to the previous question, and provide corresponding answers.'
    question_category = ' The questions can be relative position between objects, the relationship between objects, object action, object attributes, object status, object types, etc. \n'
    certain = 'Only include questions that have definite answers:(1) one can see the content in the image that the question asks about and can answer confidently;(2) one can determine confidently from the image that it is not in the image. Do not ask any question that cannot be answered confidently.\n'
    format = 'Please note that each question must refer to an object or some objects in the image, if this requirement can not be satisfied, replace this question with another.' \
             ' In addition, please follow the format strictly: Present the coordinates of every object mentioned in the question at the end of each question with the format "<object:[x1,y1,x2,y2]>". ' \
             'The order must be attached to questions and answers, like Question1 and Answer1. The coordinates referring to the answer should be put at the end of the answer and be attached to the object name, and the object name should not be put in quotation marks, for example: <the name of the object1:[x1,y1,x2,y2],the name of the object2:[x3,y3,x4,y4]>. The name of the object should be a noun phrase in relationship.' \
             ' Each round of conversation should be separated by a blank line.\n'\
             'Here is an example:\n' \
             'Question1: What is the color of the shirt of the man? <man:[0.1,0.1,0.3,0.5],shirt:[0.1,0.2,0.3,0.4]>\n' \
             'Answer1: The color is red. <shirt:[0.1,0.2,0.3,0.4]>\n\n' \
             'Question2: Where is the man sitting? <man:[0.1,0.1,0.3,0.5]>\n' \
             'Answer2: The man is sitting on a chair. <chair:[0.1,0.4,0.3,0.6]>\n\n' \
             'This example is not related to the given information.\n'
    supplement = supplement_1()
    task=st3 + str(
        times) + st4 + question_category + certain + format + supplement
    return task

def get_prompt(relationship,times=4):
    quotation = '"'
    triple_quotation = '\n""" \n'
    t2=task2(times)
    prompt=str(relationship)+triple_quotation+t2+quotation
    return prompt


def get_prompt_or(region,object,times=4):
    st1=" [{'role': 'system', 'content': "
    quotation='"'
    st2='You are an AI visual assistant observing a single image. You have information about the position of all objects in this image, represented as bounding boxes (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. You also receive several region descriptions with corresponding bounding boxes, each describing a specific region in the image. The specific information related to the image will be delimited with""" characters. \n '
    triple_quotation='""" \n'
    region_def = ' Region: including region description and corresponding coordinates. \n'
    object_def = ' Object: including object category and corresponding coordinates. \n'
    st3=' Your task is to ask a question about any object in the image and provide an answer. Then, generate '
    st4=' new questions in order, each based on the answer to the previous question, and provide corresponding answers. The questions can relate to the position between objects, the relationship between objects, object actions, object attributes, or object status.\n '
    st5='Ensure each question refers to an object or objects in the image. If a question does not satisfy this requirement, replace it with one that does. Follow this format strictly: Present the coordinates of the selected object at the end of each question using the format "<object:[x1,y1,x2,y2]>". Attach the question and answer order, like Question1 and Answer1. Attach the coordinates referring to the answer at the end, connected to the object name. Do not put the object name in quotation marks. For example: <object1:[x1,y1,x2,y2], object2:[x3,y3,x4,y4]>. '
    st6="Here is an example: \n Question1:  What is the color of the man'shirt? <man:[0.1,0.1,0.3,0.5]> \n Answer1: The color is red. <shirt:[0.1,0.2,0.3,0.4]> \n This example is not related to the given information."
    st7='}]'
    content=st2+triple_quotation+region_def+str(region)+object_def+str(object)+'\n'+triple_quotation+st3+str(times)+st4+st5+st6
    prompt=st1+quotation+content+quotation+st7
    return prompt

def send_request_to_openai(model, messages):
    # 发送请求到 OpenAI
    response = openai.ChatCompletion.create(model=model, messages=messages,temperature=0.5)
    # print(response)
    # 返回 OpenAI 的响应
    return response

def chat(content):
    st2 ='You are an AI visual assistant, and you are seeing a single image. You know several descriptions of the image, each describing a specific relationship in the image you are observing with the position of objects in this image in the form of bounding box represented as (x1, y1, x2, y2) .Please note that there may be several objects of the same category in an image, and we use "name_number" to present these objects, such as "man_1","man_2" to describe different men. '
    relationship_def = ' Relationship: the relationship between two objects (with name and coordinates) \n'
    sys_content=st2+relationship_def
    #get answer
    a1 = datetime.datetime.now()
    response = send_request_to_openai('gpt-4',
                                    [{'role': 'user', 'content': content},{'role':'system','content':sys_content}])
    answer=response.choices[0].message['content']
    a2=datetime.datetime.now()
    t=a2-a1
    print('time: {}'.format(t))
    question_tokens = response['usage']['prompt_tokens']
    # 计算回答的token数量
    answer_tokens = response['usage']['total_tokens'] - question_tokens
    # 打印响应的内容
    print("问题消耗:",question_tokens,"tokens")
    print("回答消耗:",answer_tokens,"tokens")
    return answer

