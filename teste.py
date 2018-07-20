import re

tweet = "this tweet is example #key1_key2_key3@"
new_tweet = " ".join(word.strip() for word in re.split('#|_|@', tweet))

list = ['this tweet is example #key1_key2_key3@','#renato lindo @@','https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python']


def remove_hashtag(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = " ".join(word.strip() for word in re.split('#|_|@', word))
        processed_word_list.append(limpo)

    return processed_word_list
def remove_url(word_list):
    processed_word_list = []

    for word in word_list:
        limpo = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)
        processed_word_list.append(limpo)

    return processed_word_list
print(remove_url(list))