import chatbot, crisis, emotion

def respond(text, history, past_answers):
    crisis_result = crisis.detect(text)
    if crisis_result['is_crisis']:
        return {'crisis': crisis_result, 'emotion': None, 'answer': crisis.HELP_MESSAGE}

    emotion_result = emotion.predict(text, history)
    answer = chatbot.reply(text, emotion_result, history, past_answers)
    return {'crisis': crisis_result, 'emotion': emotion_result, 'answer': answer}

if __name__ == '__main__':
    history, past_answers = [], []
    while (text := input('\n나: ').strip()) != '끝':
        result = respond(text, history, past_answers)
        history.append(text)
        past_answers.append(result['answer'])
        if result['emotion']:
            print(f"[감정: {result['emotion']['emotion']}]")
        print('챗봇:', result['answer'])
