# imports
from utils.beautify import n, lg, r, cy, ye    # 0
from train.gpt import train_gpt
from train.bigram import train_bigram


def execute():
    while True:
        print(ye+"\n===================  Train Options ==================="+n)
        print(lg+'[1] Train Bigram Model'+n)
        print(lg+'[2] Train GPT Model'+n)
        print(lg+'[3] Go Back'+n)
        
        # entering choice
        try:
            op = int(input('\nEnter your choice: '))
            if op < 1 or op > 3:
                print(r+f"Please Choose a Value between 1-3\n\n"+n)
                return
        except ValueError:
            print(r+"Please Enter Correct Value and Retry\n\n"+n)
            return

        # Q1
        # TODO: Bigram Model to be done
        if op == 1: 
            print("\nTRAINING A BIGRAM MODEL : ")
            train_bigram()
        
        # Q2
        if op == 2: 
            print("\nTRAINING A GPT-like LLM : ")
            train_gpt()

        # go back
        if op == 3: return


