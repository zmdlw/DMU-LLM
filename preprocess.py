import pandas as pd

chatbot = pd.read_csv('chatbot.csv').rename(columns={'Q': '문장', 'A': '답변'})

wellness = pd.read_excel('wellness.xlsx').rename(columns={'유저': '문장', '챗봇': '답변'})
wellness['답변'] = wellness.groupby('구분')['답변'].ffill()
wellness['감정'] = wellness['구분'].str.split('/').str[1]

symptom = pd.read_excel('symptom.xlsx')
symptom.columns = symptom.columns.str.strip()
symptom['감정'] = symptom['intent'].ffill().str.split('/').str[1]
symptom = symptom.melt(id_vars='감정', value_vars=['utterance', 'utterance(2차)'], value_name='문장')

data = pd.concat([chatbot, wellness, symptom], ignore_index=True).dropna(subset=['문장'])
data['문장'] = data['문장'].str.strip()
data['감정'] = data['감정'].replace('불안감', '불안')

crisis_words = '자살|자해를|자해하|죽고 싶|죽고싶|죽고만 싶|죽어버리고 싶|살기 싫'
is_crisis = data['문장'].str.contains(crisis_words) | data['감정'].isin(['자살충동', '자살시도', '자해'])
data['위기'] = is_crisis.astype(int)

chatbot_data = data[data['위기'] == 0].dropna(subset=['답변'])[['문장', '답변']].drop_duplicates()
emotion_data = data[data['감정'].isin(symptom['감정'])][['문장', '감정']].drop_duplicates('문장')
crisis_data = data.sort_values('위기', ascending=False)[['문장', '위기']].drop_duplicates('문장')

chatbot_data.to_csv('chatbot_data.csv', index=False, encoding='utf-8-sig')
emotion_data.to_csv('emotion_data.csv', index=False, encoding='utf-8-sig')
crisis_data.to_csv('crisis_data.csv', index=False, encoding='utf-8-sig')

print('대화 데이터:', len(chatbot_data))
print('감정 데이터:', len(emotion_data))
print('위기 데이터:', len(crisis_data), '(위기', crisis_data['위기'].sum(), ')')