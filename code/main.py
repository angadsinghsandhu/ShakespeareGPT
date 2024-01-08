# imports
from utils.beautify import n, lg, r, cy, ye    # 0
from utils.beautify import init                # 0

import train               # 1, 2
import generate

# last choice num
limit = 2

## logic ##
init()

# inf loop
while True:
    # display choices
    print(cy+"================================================"+n)
    print(cy+"================================================\n"+n)
    print(lg+'[-1] Quit'+n)
    print(lg+'[0] Banner + Clear Terminal'+n)
    print(lg+'[1] Train Model'+n)
    print(lg+'[2] Run Model'+n)
    
    # entering choice
    try:
        ch = int(input('\nEnter your choice: '))
        if ch < -1 or ch > limit:
            print(r+f"Please Choose a Value between 0-{limit}\n\n"+n)
            continue
    except ValueError:
        print(r+"Please Enter Correct Value and Retry\n\n"+n)
        continue

    # exit from program
    if ch == -1:
        init()
        print(f'\n{r} [i] choice {ch} complete !!\n')
        exit()
    
    # to clear terminal and show banner
    elif ch == 0: init()

    # create/recreate a session file for a new number 
    elif ch == 1: train.menu.execute()
        
    # Add all the numbers to a specified group
    elif ch == 2: generate.menu.execute()

    print(f'\n{r} [i] choice {ch} complete !!\n')


# TODO: Complete Bigram methods
def train_bigram():
    pass

def generate_bigram():
    pass
