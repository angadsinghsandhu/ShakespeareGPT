# imports
from utils.beautify import n, lg, r, cy, ye    # 0
from generate.gen_gpt import generate_gpt
from generate.gen_bigram import generate_bigram
from generate.gen_base import generate_base

def execute():
    while True:
        print(ye+"\n===================  Generate Options ==================="+n)
        print(lg+'[1] Generate from Base Model'+n)
        print(lg+'[2] Generate from Bigram Model'+n)
        print(lg+'[3] Generate from GPT Model'+n)
        print(lg+'[4] Go Back'+n)
        
        # entering choice
        try:
            op = int(input('\nEnter your choice: '))
            if op < 1 or op > 4:
                print(r+f"Please Choose a Value between 1-4\n\n"+n)
                return
        except ValueError:
            print(r+"Please Enter Correct Value and Retry\n\n"+n)
            return

        if op == 1: 
            print("\nGENERATING FROM BASE MODEL : ")
            generate_base()
        
        # TODO: add Bigram model generation
        if op == 2: 
            print("\nGENERATING FROM BIGRAM MODEL : ")
            generate_bigram()

        # Q3
        if op == 3: 
            print("\nGENERATING FROM GPT-like LLM : ")
            generate_gpt()

        # go back
        if op == 4: return