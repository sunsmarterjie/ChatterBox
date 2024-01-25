from curses.ascii import isdigit


def txt2json(txt):
    a=0
    q=0
    cnt=0
    conversation=[]
    txt_list=txt.split('\n')
    for line in txt_list:
        if 'Question' in line or 'question' in line:
            question=line.strip('\n')
            if ': ' in question:
                q_order,question=question.split(': ')[0],question.split(': ')[-1]
                q_order=q_order[-1]
                q=1
            elif ':' in question:
                q_order,question=question.split(':')[0],question.split(':')[-1]
                q_order = q_order[-1]
                q=1
            cnt+=1
        elif 'Answer' in line or 'answer' in line:
            answer=line.strip('\n')
            if ': ' in answer:
                a_order, answer = answer.split(': ')[0], answer.split(': ')[-1]
                a_order = a_order[-1]
                a = 1
            elif ':' in answer:
                a_order, answer = answer.split(':')[0], answer.split(':')[-1]
                a_order = a_order[-1]
                a = 1
            cnt += 1
        else:
            pass
        if q==1 and a==1:
            if isdigit(a_order) and isdigit(q_order):
                if a_order ==q_order:
                    qd={'from':'human','value':question}
                    ad={'from':'gpt','value':answer}
                    conversation.append(qd)
                    conversation.append(ad)
            else:
                qd = {'from': 'human', 'value': question}
                ad = {'from': 'gpt', 'value': answer}
                conversation.append(qd)
                conversation.append(ad)
            q = 0
            a = 0
        elif cnt==2:
            q = 0
            a = 0
        else:
            pass
    return conversation