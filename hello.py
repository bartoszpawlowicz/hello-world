class Message:
    """ This class handles a message
    
        At this moment only a content is handles cause I don't yet have an idea what else to add
    """
    def __init__(self, content="Hello World"):
        self.content = content


if __name__ == '__main__':
    first_message = Message()
    print(first_message.content)
