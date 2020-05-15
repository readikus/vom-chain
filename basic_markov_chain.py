from nltk.tokenize import word_tokenize
import json
from random import randrange

#import nltk
#nltk.download('punkt')

# this is BASIC - not strictly a probability model, but will have the same affecr
class TransititionModel():
    def __init__(self):
        self.words = []
        # @todo: change to use a sparse matrix
        self.transitions = {}

    # either hash it, or 
    def add(self, s):
        tokens = ('STARTOFSEQUENCE ' + s + ' ENDOFSEQUENCE').split()
        for i in range(len(tokens) - 1):
            a = tokens[i]
            b = tokens[i + 1]
            self.update_transitions(a, b)

    # records a transition from a -> b
    def update_transitions(self, a, b):
        if a not in self.transitions:
            self.transitions[a] = []
        self.transitions[a].append(b)

    def generate(self):
        random_sequence = ''
        current_symbol = self.get_next_symbol(u'STARTOFSEQUENCE')
        while current_symbol != u'ENDOFSEQUENCE':
            random_sequence += ' ' + current_symbol
            current_symbol = self.get_next_symbol(current_symbol)
            #print('current_symbol', current_symbol)
        return random_sequence

    def get_next_symbol(self, x_t):
        return self.transitions[x_t][randrange(0, len(self.transitions[x_t]))]

# load the twitter data
def load_tweets(file_name):
    with open(file_name) as data_file:
        data = json.load(data_file)
        return data

def build_transition_matrix(tweets):
    model = TransititionModel()
    for tweet in tweets:
        model.add(tweet)
    return model

tweets = load_tweets('./data/elonmusk.json')
generator = build_transition_matrix(tweets)
print(generator.generate())

print('------')
print(generator.generate())
print('------')
print(generator.generate())
print('------')
print(generator.generate())
print('------')
