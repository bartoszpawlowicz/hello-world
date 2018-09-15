class Message:
    def __init__(self, content="Hello World"):
        self.content = content


def anagram_check(s1, s2):
    list1 = list(s1)
    list2 = list(s2)

    list1 = sorted(list1)
    list2 = sorted(list2)

    return list1 == list2


if __name__ == '__main__':
    first_message = Message()
    second_message = Message("World Hello")
    decision = anagram_check(first_message.content, second_message.content)
    print(first_message.content + " and " + second_message.content + " are anagrams: " + str(decision))
