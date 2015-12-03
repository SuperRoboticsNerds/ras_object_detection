
import numpy as np 
import os, sys
import re

# step to run the file:
# python transform_data.py  <file_name.txt> <out_file>  

def readFile(filename):
	pass


def createNpArray(text):
	feat_data = np.array(text.split(','), dtype='uint8')
	feat_data = np.reshape(feat_data, (feat_data.shape[0] / 3, 3))
	print feat_data.shape
	return feat_data




if __name__ == "__main__":

	if len(sys.argv) < 2:
		raise Exception("Not enough input") 
	
	filename = sys.argv[1]
	out_file = sys.argv[2]

	# filename = 'green_dark_cube.txt'
	with open(filename) as f:
		text = f.read().replace('\n', '')
		text_useful = re.findall(r'\[([^]]+)\]', text, re.MULTILINE)
		raw_text = ', '.join(text_useful)
		# print raw_text
		# print text_useful
		data = createNpArray(raw_text)
		print data.shape
		print data
		np.savez(out_file, data)



	# lines = [l.rstrip('\n').rstrip() for l in lines]

	# # print len(lines)

	# feat_mult = [l.split(';') for l in lines]
	# print feat_mult[:5]

	# comb_feat_raw = []
	# cf = [comb_feat_raw.extend(el) for el in feat_mult] 
	# print cf[:5]


