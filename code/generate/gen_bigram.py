import torch
from train.bigram import BigramLanguageModel, decode

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------

def generate_bigram():
    # loading model
    m = BigramLanguageModel()
    m.load_state_dict(torch.load("model/bigram_state_2023_05_07-23_24.pt", map_location=device))
    m.eval()

    # generate from the model
    # initial context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    # open('./data/sample_output_bigram.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))