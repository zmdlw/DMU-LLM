import os
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForSeq2Seq

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, 'models', 'kogpt2-chatbot')
MAX_LEN = 64

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
tokenizer.add_special_tokens({'additional_special_tokens': ['<usr>', '<sys>']})
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.resize_token_embeddings(len(tokenizer))

def make_example(question, answer):
    question_ids = tokenizer.encode('<usr>' + question + '<sys>')[-(MAX_LEN // 2):]
    answer_ids = tokenizer.encode(answer) + [tokenizer.eos_token_id]
    return {'input_ids': (question_ids + answer_ids)[:MAX_LEN], 'labels': ([-100] * len(question_ids) + answer_ids)[:MAX_LEN]}

data = pd.read_csv(os.path.join(ROOT, 'data', 'chatbot_data.csv'))
examples = [make_example(q, a) for q, a in zip(data['문장'], data['답변'])]

trainer = Trainer(
    model=model,
    args=TrainingArguments(OUTPUT, num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=4, learning_rate=5e-5, logging_steps=100, save_strategy='no', report_to='none', use_cpu=True),
    train_dataset=examples,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

trainer.train()
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)