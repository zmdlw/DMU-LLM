import os, torch
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, 'models', 'kogpt2-chatbot')

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LEN = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
tokenizer.add_special_tokens({'additional_special_tokens': ['<usr>', '<sys>']})
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.resize_token_embeddings(len(tokenizer)) 
model.to(device)

def make_example(question, answer):
    question_ids = tokenizer.encode('<usr>' + question + '<sys>')[-(MAX_LEN // 2):]
    answer_ids = tokenizer.encode(answer) + [tokenizer.eos_token_id]
    input_ids = (question_ids + answer_ids)[:MAX_LEN]
    labels = ([-100] * len(question_ids) + answer_ids)[:MAX_LEN]
    attention_mask = [1] * len(input_ids)
    empty = MAX_LEN - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * empty
    labels = labels + [-100] * empty
    attention_mask = attention_mask + [0] * empty
    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

data = pd.read_csv(os.path.join(ROOT, 'data', 'chatbot_data.csv'))
examples = []
for question, answer in zip(data['문장'], data['답변']):
    examples.append(make_example(question, answer))

batches = torch.utils.data.DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for input_ids, attention_mask, labels in batches:
        output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                       labels=labels.to(device))
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        total_loss += output.loss.item()
    print(f'epoch: {epoch + 1}, loss: {total_loss / len(batches):.4f}')
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print(OUTPUT)