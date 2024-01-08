from colorama import Fore
import os, json

n = Fore.RESET
lg = Fore.LIGHTGREEN_EX
r = Fore.RED
w = Fore.WHITE
cy = Fore.CYAN
ye = Fore.YELLOW
colors = [lg, r, w, cy, ye]


info = json.load(open('./env.json'))
version = info['version']
# authors = info['authors']

# initial function to clear terminal and display the banner
def init():
    clr()
    banner()

# clear the terminal
def clr():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# display the script banner
def banner():
    import random
    # fancy logo
    b = [
        "██████╗░██╗░░░░░██╗░░██╗███████╗   ██╗░░░░░██╗░░░░░███╗░░░███╗",
        "██╔══██╗██║░░░░░██║░░██║██╔════╝   ██║░░░░░██║░░░░░████╗░████║",
        "██████╔╝██║░░░░░███████║█████╗░░   ██║░░░░░██║░░░░░██╔████╔██║",
        "██╔══██╗██║░░░░░██╔══██║██╔══╝░░   ██║░░░░░██║░░░░░██║╚██╔╝██║",
        "██║░░██║███████╗██║░░██║██║░░░░░   ███████╗███████╗██║░╚═╝░██║",
        "╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚═╝░░░░░   ╚══════╝╚══════╝╚═╝░░░░░╚═╝",
    ]
    
    for char in b:
        print(f'{random.choice(colors)}{char}{n}')
    #print('=============TopX X Reaper==============')
    # x, y, z = authors
    # print(f'   Version: {version}  | Author TG: {x} X {y} X {z}{n}\n')