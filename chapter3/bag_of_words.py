import collections, re
texts = ['John likes to watch movies. Mary likes too.', 'John also likes to watch football games.']
bagsofwords = [collections.Counter(re.findall(r'\w+', txt)) for txt in texts]
print(bagsofwords[0])
print(bagsofwords[1])
sumbags = sum(bagsofwords, collections.Counter())
print(sumbags)
