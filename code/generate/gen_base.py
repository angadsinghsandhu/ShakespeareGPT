import random
from train.gpt import decode

def generate_base():
    rand_list=[]
    n=50
    for i in range(n):
        rand_list.append(random.randint(0,64))

    print(decode(rand_list))