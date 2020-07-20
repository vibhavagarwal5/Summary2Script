# Using readlines() 
file1 = open('Aliens.txt', 'r') 
Lines = file1.readlines() 



count = 0
# Strips the newline character 
chars = []
spaces = list()

i = 0
for line in Lines: 
    count=0
    flag = 0
    flag2 = 1
    # print (line)
    for i in line:
        if(i.isspace() and flag == 0):
            count=count+1
        if(i.isalpha()):
            spaces.append(count)
            flag = 1
            break
    if(count == 35):
        chars.append(line.strip())
    # print("The number of blank spaces is: ",count)
print (chars)


dialogs = dict.fromkeys(chars, [])

dialog = str()
for i in range(len(Lines)):
    count = 0
    flag = 0
    dialog = str()
    for j in Lines[i]:
        if(j.isspace() and flag == 0):
            count=count+1
        if(j.isalpha()):
            flag = 1
    
    if(count == 35):
        char = Lines[i].strip()
        complete = str()
        while(spaces[i] != 20):
            i = i + 1
        while(i < len(spaces) and spaces[i] == 20 ):
            complete = complete + Lines[i].strip() + " " 
            i = i + 1
        dialogs[char].append(complete)
for key, value in dialogs.items() :
    print (key, value)