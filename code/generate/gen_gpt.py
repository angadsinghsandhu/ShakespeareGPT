import torch
from train.gpt import GPTLanguageModel, decode

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------

def generate_gpt():
    # loading model
    m = GPTLanguageModel()
    m.load_state_dict(torch.load("model/gpt_state.pt", map_location=device))
    m.eval()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    # open('./data/sample_output_gpt.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))