from nltk.tokenize import word_tokenize
import json
from random import randrange
import numpy
from fractions import Fraction as frac
from random import choices
from nltk.util import ngrams
from collections import Counter

# this is BASIC - not strictly a probability model, but will have the same affecr
class MarkovModel():
    
    def __init__(self):
        self.words = []
        # @todo: change to use a sparse matrix - but get 100% test coverage first!
        self.transitions = {}

    def tokenize(self, doc, n = 2):
        sequence = doc.split(' ')
        return list(ngrams(sequence, n, pad_left=True, pad_right=True, left_pad_symbol='<S>', right_pad_symbol='</S>'))

    def build_vocab(self, docs, n):

        if n < 2:
            raise Error('n must be 2 or more')
        words = set()
        for i in range(1, n + 1):
            for doc in docs:
                tokens = self.tokenize(doc, i)
                for token in tokens:
                    # add "history" if more than 1 symbol
                    if i > 1:
                        words.add(tuple(token[0:-1]))

                    words.add(tuple((token[-1],)))
        return sorted(list(words))

    def train(self, docs, n):
        self.n = n
        self.words = self.build_vocab(docs, n)
        self.transitions = numpy.zeros(shape=(len(self.words), len(self.words)))
        self.population = range(0, len(self.words))

        for doc in docs:
            tokens = self.tokenize(doc, n)
            # record each transition
            for token in tokens:
                a = self.encode_word(tuple(token[0:-1]))
                b = self.encode_word(tuple((token[-1],)))
                self.transitions[a][b] += 1

        # transform to probabilities
        self.transitions = self.transitions.astype('object')

        for i in range(len(self.transitions)):
            # calc row total
            total = numpy.sum(self.transitions[i])
            # if no emitting states, it is a final state - so only ever goes to itself
            # to maintain a correct probability distro
            # @todo - set fraction at this bit...
            if total == 0:
                self.transitions[i][i] = 1
                total = 1

            for j in range(len(self.transitions[i])):
                if self.transitions[i][j] != 0:
                    self.transitions[i, j] = frac(int(self.transitions[i][j]), int(total))

    def encode_word(self, word):
        return self.words.index(word)

    # records a transition from a -> b
    def update_transitions(self, a, b):
        if a not in self.transitions:
            self.transitions[a] = []
        self.transitions[a].append(b)

    def generate(self):

        # generate a start sequence (i.e. <S> for (n-1))
        start_symbol = tuple((['<S>' for x in range(self.n - 1)]))
        start_index = self.encode_word(start_symbol)
        
        # seed the sequence from the start
        current_index = self.generate_next_symbol(start_index)
        last_n = start_symbol[1:] + self.words[current_index]

        reached_stop_state = False      
        random_sequence = []

        while not reached_stop_state:

            i = self.encode_word(last_n)
            #prev_index = current_index
            random_sequence.append(str(self.words[current_index][-1]))

            current_index = self.generate_next_symbol(i)
            last_n = last_n[1:] + self.words[current_index]
            print(random_sequence)

            reached_stop_state = self.words[current_index][0] == '</S>'

           # exit()
            #print('moving from ' + str(prev_index) + '(' + str(self.words[prev_index]) + ') to ' + str(current_index) + '(' + str(self.words[current_index]) + ')' )
        return (' ').join(random_sequence)

    def generate_next_symbol(self, i):

        # current symbol index
        #i = self.words.index(current_symbol
        # 
        # )
        #print('i: ' + str(i))
        #print(self.transitions[i])

        return choices(self.population, self.transitions[i]).pop()

# load the twitter data
def load_tweets(file_name):
    with open(file_name) as data_file:
        data = json.load(data_file)
        return data

def build_transition_matrix(tweets):
    model = MarkovModel()
    model.train(tweets, 3)
    return model



def main():

    tweets = ['a b c', 'a b c f', 'a b c d', 'a b c f', 'b c f', 'a c f', 'a', 'b c d', 'a c q', 'a c z', 'a c r', 'a c r s']
    tweets = ['a b', 'a c']


    tweets = load_tweets('./data/trump.json')


    s  = 'alan bob colin went to the shop for chips'
    #ng = ngrams(s.split(' '), 2)
    #print(list(ng))
    #exit()

    #markov_model = MarkovModel()

    #markov_model.tokenize(s, 3)

    model = build_transition_matrix(tweets)
    print('model')
    print(model)
    print(model.generate())

    print('------')
    print(model.generate())
    print('------')
    print(model.generate())
    print('------')
    print(model.generate())
    print('------')

if __name__ == '__main__':
    main()
