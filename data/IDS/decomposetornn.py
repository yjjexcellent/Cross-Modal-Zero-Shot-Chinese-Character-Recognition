s = open('decompose_27533_benchmark.txt', 'r', encoding='utf-8')

stru_list = ['⿱', '⿰', '⿺', '⿹', '⿸', '⿶', '⿷', '⿵', '⿴', '⿻', '⿳', '⿲']

content = s.read()

i = content.find('⿲')
while i != -1:
    if content[i+2] not in stru_list:
        content = content[:i] + '⿰' + content[i+1:]
        if content[i+2] != '&':
            content = content[:i+4] + '⿰ ' + content[i+4:]
        else:
            content = content[:i+13] + '⿰ ' + content[i+13:]
    elif content[i+4] not in stru_list and content[i+2] != '⿳' and content[i+2] != '⿲':
        if content[i+4] != '&' and content[i+6] not in stru_list:
            content = content[:i] + '⿰' + content[i+1:]
            if content[i+6] != '&':
                content = content[:i+8] + '⿰ ' + content[i+8:]
            else:
                content = content[:i+17] + '⿰ ' + content[i+17:]
        elif content[i+4] == '&' and content[i+15] not in stru_list:
            content = content[:i] + '⿰' + content[i+1:]
            if content[i+15] != '&':
                content = content[:i+17] + '⿰ ' + content[i+17:]
            else:
                content = content[:i+26] + '⿰ ' + content[i+26:]
    i = content.find('⿲', i + 1)

i = content.find('⿳')
while i != -1:
    if content[i+2] not in stru_list:
        content = content[:i] + '⿱' + content[i+1:]
        if content[i+2] != '&':
            content = content[:i+4] + '⿱ ' + content[i+4:]
        else:
            content = content[:i+13] + '⿱ ' + content[i+13:]
    elif content[i+4] not in stru_list and content[i+2] != '⿳' and content[i+2] != '⿲':
        if content[i+4] != '&' and content[i+6] not in stru_list:
            content = content[:i] + '⿱' + content[i+1:]
            if content[i+6] != '&':
                content = content[:i+8] + '⿱ ' + content[i+8:]
            else:
                content = content[:i+17] + '⿱ ' + content[i+17:]
        elif content[i+4] == '&' and content[i+15] not in stru_list:
            content = content[:i] + '⿱' + content[i+1:]
            if content[i+15] != '&':
                content = content[:i+17] + '⿱ ' + content[i+17:]
            else:
                content = content[:i+26] + '⿱ ' + content[i+26:]
    i = content.find('⿳', i + 1)

f = open('decompose_27533_benchmark_rnn.txt', 'w+', encoding='utf-8')
f.write(content)
