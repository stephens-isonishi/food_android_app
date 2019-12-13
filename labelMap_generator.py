import os, sys, csv
from glob import glob

def createClassNames(directory):
	class_names = [x.split("/")[-1] for x in glob(directory)]
	class_names = sorted(class_names)
	name_id_hashmap = dict(zip(class_names, range(len(class_names))))
	return name_id_hashmap

def main():
	labels = createClassNames('../training_data/*')
	with open('labelMap.csv', mode='w', newline="") as csv_file:
		writer = csv.writer(csv_file)
		for key, value in labels.items():
			writer.writerow([value, key])

if __name__ == '__main__':
	main()