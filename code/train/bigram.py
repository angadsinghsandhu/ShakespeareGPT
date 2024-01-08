# imports
import wget
import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime as dt

# hyperparameters
batch_size = 32         # sequences running in parallel (B)
block_size = 8          # max sequence length (T)
# vocab_size = ...      # retrieved drectly from number of unique charaters in dataset (C)
max_iters = 3000        # iterations to train model
eval_interal = 300      # interval in training max_iters to check loss
eval_iters = 200        # number of batches to average-over to minimize training noise
learning_rate = 1e-2    # learning rate of optimizer
# to run with GPU on collab: Runtime > Change Runtime Type > GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------

torch.manual_seed(1337)

DATA_PATH = "./data/tinyshakespeare.txt"

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# set of unique characters text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating mappings
stoi = {ch:idx for idx, ch in enumerate(chars)}
itos = {idx:ch for idx, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[idx] for idx in l])

# splitting training and validation data
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))    # 90% train-val split
train_data = data[:split]
val_data = data[split:]

# loading data into batches
def get_batch(split_type):
    # generate small batch data of inputs x and labels y
    data = train_data if split_type == 'train' else val_data

    # getting array of positions/indexes of the first element of each block
    # array is of size batch_size
    # random integers between 0 to (dataset size - the block size)
    # return the array in shape (batch_size, ) i.e. 1D addar of length batch_size
    x_idx = torch.randint(len(data)-block_size, (batch_size, ))

    # generating block sequences from indexes
    # creating input tensor of shape (B, T) : (batch_size, block_size)
    X = torch.stack([data[ i : i+block_size ] for i in x_idx])

    # generating correct return sequences of each block using next element in data
    # creating input tensor of shape (B, T) : (batch_size, block_size)
    Y = torch.stack([data[ i+1 : i+block_size+1 ] for i in x_idx])

    # sending inputs and labels to appropriate GPU/CPU device
    X, Y = X.to(device), Y.to(device)
    
    return X, Y

@torch.no_grad() # Context-manager that disabled gradient calculation
def estimate_loss(model):
    out = {}    # output data structure containing loss values

    # setting our custome model class to evaluating mode
    model.eval()

    # looping through data splits
    for split in ['train', 'val']:
        # initializing loss vector of size `eval_iters`
        losses = torch.zeros(eval_iters)

        # looping through each evaluation
        for k in range(eval_iters):
            # getting batch of data
            X, Y = get_batch(split)

            # train model and get output data
            logits, loss = model(X, Y)

            # add loss to main loss vector
            losses[k] = loss.item()

        # averaging and appending to output
        out[split] = losses.mean()

    # setting our custom model class back to training mode
    model.train()

    # returning output
    return out

# creating simple bigram model
class BigramLanguageModel(nn.Module):

    # this method is called when the model variabe is initialized
    def __init__(self):
        # calling parent nn.Module class
        super().__init__()

        # creating lookup table of logits for each token
        # creating embedding table of (size of the dictionary of embeddings) by (the size of each embedding vector)
        # example: for an embedding with 1,000 tokens each of 200 dimentions, the table would be Embedding(1000, 200)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # this method is called when the initialize class variable is called like a function
    # the inputs to this are inputs (idx) and targets (labels)
    # where the labels are optional for when we want to calculate the loss 
    def forward(self, idx, targets=None):

        # both inputs and labels are of size (B, T)
        # adding embedings to each token in the input sequence as another dimention 'C'
        logits = self.token_embedding_table(idx)

        # calculate loss if 
        if targets is None:
            loss = None
        else:
            # getting Tensor shape dimentions
            B, T, C = logits.shape

            # reshaping logits and targets to align with cross entropy syntactically
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # calculate loss using loss function 
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss
    
    # This method is caled after training to take the context as input (idx) 
    # and generate till a limited number of charaters, defined by max_new_tokens
    def generate(self, idx, max_new_tokens):

        # looping through the number of tokens o be generated
        for _ in range(max_new_tokens):

            # getting the predictions by calling the class itself i.e. forward()
            # here input (idx) is again in the shape of (B, T)
            logits, loss = self(idx)

            # now we take out the last block i.e. last token in the sequence
            logits = logits[:, -1, :]       # logits becomes of shape (B, C)

            # apply softmax to embeddings to convert into a probability distribution
            probs = F.softmax(logits, dim = -1)

            # TODO : idk what this does
            # sampling from distribution the next charater
            idx_next = torch.multinomial(probs, num_samples=1)

            # adding sample to input to generate next character
            idx = torch.cat((idx, idx_next), dim=1)     # (B, T+1) 

        return idx

def train_bigram():
    # creating model instance
    model = BigramLanguageModel()
    m = model.to(device=device)     # sending model to GPU/CPU

    # creating PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training the Model
    for iter in range(max_iters):

        # checking loss in every eval_iterval steps
        if iter % eval_interal == 0:
            losses = estimate_loss(m)
            # losses = estimate_loss(model)
            print(f"step {iter} | train loss = {losses['train']:.4f} | val loss = {losses['val']:.4f}")

        # sample batch of data
        X, Y = get_batch('train')

        # evaluate loss
        logits, loss  = m(X, Y)

        # set all gradients to zero
        optimizer.zero_grad()

        # backwar propagation step
        loss.backward()

        # perform single optimization step
        optimizer.step()

    # getting current time string
    timestamp = dt.now().strftime("_%Y_%m_%d-%H_%M")

    # saving model
    torch.save(m.state_dict(), f"model/bigram_state{timestamp}.pt")