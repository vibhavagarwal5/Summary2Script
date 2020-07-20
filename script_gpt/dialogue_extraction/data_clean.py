file1 = open('Aliens.txt', 'r+') 
Lines = file1.readlines() 

for line in Lines:
    flag = 0
    for i in line:
        if(i.isalpha()):
            flag = 1
    if(flag == 1):
        file1.write(line)