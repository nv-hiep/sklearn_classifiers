import string
import numpy   as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(
	               zip( range(1, len(characters) + 1),   characters )
	              )

print( characters )
print( token_index )

max_length = 50
results = np.zeros(
	               (
	               	len(samples),
	               	max_length,
	               	max(token_index.keys()) + 1
	               	)
	              )

for (i, sample) in enumerate(samples):
	for (j, character) in enumerate(sample):
		index                = [k for k,v in token_index.items() if v == character][0]
		results[i, j, index] = 1.
	# End - for
# End - for

# print( results )