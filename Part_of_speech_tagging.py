import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    return tagged

def print_pos_tags(tagged):
    for word, tag in tagged:
        print(f'{word}: {tag}')

if __name__ == "__main__":
    sentence = "Im student of Artificial Inteligence"
    tagged = pos_tagging(sentence)
    print_pos_tags(tagged)
