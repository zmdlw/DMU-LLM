import chatbot

if __name__ == '__main__':
    while (text := input('\n나: ').strip()) != '끝':
        print('챗봇:', chatbot.generate(text))
