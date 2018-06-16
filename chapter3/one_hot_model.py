# define input string
data = 'the quick brown fox jumped over the lazy dog'
consecutive_words = data.split()
print(data)

# construct the dictionary
all_words = list(set(consecutive_words))

# define a mapping of word to integers
word_to_int = dict((w, i) for i, w in enumerate(all_words))
int_to_word = dict((i, w) for i, w in enumerate(all_words))
# integer encode input data
integer_encoded = [word_to_int[w] for w in consecutive_words]

# one hot encode
onehot_encoded = list()
for value in integer_encoded:
    letter = [0 for _ in range(len(all_words))]
    letter[value] = 1
    onehot_encoded.append(letter)
_ = [print(x) for x in onehot_encoded]

def argmax(vector):
    # since vector is actually a list and its one hot encoding hence the
    # maximum value is always 1
    return vector.index(1)

# invert encoding
inverted = int_to_word[argmax(onehot_encoded[0])]
print(inverted)
