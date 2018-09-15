class Message:
    def __init__(self, content="Hello World"):
        self.content = content


if __name__ == '__main__':
    first_message = Message()
    print(first_message.content)
