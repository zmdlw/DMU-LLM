import os, re, requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, GPT2LMHeadModel

ROOT = os.path.dirname(os.path.abspath(__file__))
GPT = os.path.join(ROOT, 'models', 'kogpt2-chatbot')
load_dotenv(os.path.join(ROOT, '.env'))
GEMINI_KEY = os.environ['GEMINI_API_KEY']
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent'

BAD_WORDS = ['자살', '자해', '죽어', '죽여', '죽을', '약을 드세요', '약 드세요', '약을 먹으세요', '약을 끊으세요', '진단', '병이에요', '바보', '멍청']
SAFE_ANSWER = '그런 마음이 드셨군요. 조금 더 이야기해 주실 수 있을까요?'

RULES = '''너는 따뜻한 한국어 심리 상담 챗봇이야. 답변 하나만 써.
- 사용자가 지금 말한 구체적인 상황을 짚으며 공감하는 한 문장으로 시작
- 답변 초안이 상황에 맞으면 살리고, 안 맞으면 버려
- 상담 방향 중 상황에 더 맞는 하나를 골라, 예시를 상황에 맞게 바꾼 질문 하나로 끝내
- 이전에 챗봇이 한 질문은 반복하지 마
- 2~3문장, 150자 안팎, 한 문단, 부드러운 존댓말, 진단이나 약 이야기 금지, 답변 문장만 출력'''

CBT = {}
for line in open(os.path.join(ROOT, 'cbt_guide.txt'), encoding='utf-8'):
    emotion, technique, example = line.strip().split('|')
    CBT.setdefault(emotion, []).append((technique, example))

tokenizer = AutoTokenizer.from_pretrained(GPT)
gpt_model = GPT2LMHeadModel.from_pretrained(GPT)
GPT_SETTINGS = dict(max_new_tokens=48, do_sample=True, top_p=0.9, temperature=0.8, repetition_penalty=1.2,
                    no_repeat_ngram_size=3, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

def shorten(text):
    sentences = re.findall(r'[^.!?]+[.!?]+', ' '.join(text.split())) or [text]
    if len(sentences) > 3:
        sentences = sentences[:2] + sentences[-1:]
    return ' '.join(s.strip() for s in sentences)

def generate(text):
    ids = tokenizer.encode(f'<usr>{text}<sys>', return_tensors='pt')
    output = gpt_model.generate(ids, **GPT_SETTINGS)
    return shorten(tokenizer.decode(output[0][len(ids[0]):], skip_special_tokens=True))

def polish(text, emotion_result, draft, history, past_answers):
    emotions = [emotion_result['emotion'], emotion_result['second_emotion']]
    directions = [f'{e}: {technique} (예시: {example})' for e in emotions for technique, example in CBT.get(e, [])]
    prompt = '\n'.join([
        f"[이전에 사용자가 한 말] {' / '.join(history[-3:]) or '없음'}",
        f"[이전에 챗봇이 한 말] {' / '.join(past_answers[-2:]) or '없음'}",
        f'[지금 사용자가 한 말] {text}',
        f'[답변 초안] {draft}',
        '[상담 방향]', *directions,
    ])
    body = {'system_instruction': {'parts': [{'text': RULES}]}, 'contents': [{'parts': [{'text': prompt}]}]}
    response = requests.post(GEMINI_URL, headers={'x-goog-api-key': GEMINI_KEY}, json=body, timeout=30)
    return shorten(response.json()['candidates'][0]['content']['parts'][0]['text'])

def reply(text, emotion_result, history, past_answers):
    draft = generate(text)
    answer = polish(text, emotion_result, draft, history, past_answers)
    if any(word in answer for word in BAD_WORDS):
        return SAFE_ANSWER
    return answer