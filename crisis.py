import os, pickle
import pandas as pd
from sklearn import feature_extraction, linear_model, metrics, model_selection
from nlp import DATA, MODEL, get_words

MODEL = os.path.join(MODEL, 'crisis.pkl')

DANGER_WORDS = ['자살', '자해', '죽고 싶', '죽고싶', '죽고만 싶', '죽어버리고 싶', '죽어 버릴', '죽어버릴', '살기 싫', '살기싫', '사라지고 싶', '목숨을 끊', '목을 매', '뛰어내리', '손목을 긋', '유서', '극단적 선택', '세상을 떠나고 싶']

HELP_MESSAGE = "많이 힘드시군요. 지금 혼자 견디지 않으셔도 됩니다. 자살예방상담전화 109, 정신건강위기상담전화 1577-0199에서 24시간 이야기를 들어드려요. 위급한 상황이라면 112 또는 119에 바로 연락해 주세요."

def train():
    data = pd.read_csv(os.path.join(DATA, 'crisis_data.csv'))
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data['문장'], data['위기'], test_size = 0.2, random_state = 42)
    vec = feature_extraction.text.TfidfVectorizer(tokenizer = get_words, token_pattern = None, ngram_range = (1, 2), min_df = 2)
    clf = linear_model.LogisticRegression(class_weight = 'balanced', max_iter = 1000)
    clf.fit(vec.fit_transform(x_train), y_train)

    print(metrics.classification_report(y_test, clf.predict(vec.transform(x_test))))

    os.makedirs(os.path.dirname(MODEL), exist_ok = True)
    pickle.dump((vec, clf), open(MODEL, 'wb'))

model = None

def detect(text):
    global model
    for word in DANGER_WORDS:
        if word in text:
            return {'is_crisis': True, 'probability': 1.0, 'reason': '위험 단어: ' + word}
    if model is None:
        model = pickle.load(open(MODEL, 'rb'))
    vec, clf = model
    prob = clf.predict_proba(vec.transform([text]))[0][1]
    return {'is_crisis': prob >= 0.8, 'probability': prob, 'reason': '문맥 분석'}
if __name__ == '__main__':
    train()