import datetime
import json
import os
import openai

openai.api_base = 'https://api.ngapi.top/v1'#v1
openai.api_key='sk-rUCYa5a7mdd2DE8A6fBe7337Ec0945D29b3f785716807eDd'

def get_polish_prompt(response):
    require='You are an AI writer. You get a conversation which is about a human ask question about a photo and an AI assistant answer the question based on the photo. What you need to do is change the conversation in a more natural way while keeping the bounding boxes of the objects in the  conversation. Here is the conversation:'
    sup='Do not output the unnecessary words like "absolutely".Also, do not change the meaning of sentence, and do not delete any useful information, especially for the objects and bounding boxes.'
    prompt=require+str(response)+sup
    return prompt


def send_request_to_openai(model, messages):
    # 发送请求到 OpenAI
    response = openai.ChatCompletion.create(model=model, messages=messages,temperature=0.8)
    # print(response)
    # 返回 OpenAI 的响应
    return response

def polish(content):
    #get answer
    a1 = datetime.datetime.now()
    response = send_request_to_openai('gpt-3.5-turbo',
                                    [{'role': 'user', 'content': content}])
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