import json
import jieba


__author__ = "Singularity Point"
__version__ = "1.0.0"


def read_file(filename):

	with open(filename) as f:
		senti_list = json.load(f)

	return senti_list


POSITIVE = read_file("positive.json")
NEGATIVE = read_file("negative.json")
PRIVATIVE = read_file("privative.json")
DEGREE = read_file("degree.json")


def calc(sentence):
	
	token = jieba.cut(sentence)

	total = 0
	temp_count = 1

	for word in token:
		if word in POSITIVE:
			total += temp_count
			temp_count = 1
		elif word in NEGATIVE:
			temp_count *= -1
			total += temp_count
			temp_count = 1
		elif word in PRIVATIVE:
			temp_count *= -1
		else:
			for index, degree_list in enumerate(DEGREE):
				if word in degree_list:
					temp_count *= (index + 1)

	return total

