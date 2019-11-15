# converting from keras file to tflite file using tflite converte
import tensorflow.lite as lite
import os
import argparse


def main(args):
	file = args.filename
	if file == 'null':
		print('invalid file')
		exit(1)
	converter = lite.TFLiteConverter.from_keras_model_file(file)
	tflite_model = converter.convert()
	open('graph.tflite','wb').write(tflite_model) #graph.lite not tflite
	print('conversion successful!')


if __name__ == '__main__':
	args = argparse.ArgumentParser()
	args.add_argument('--filename', '-f', default='null')
	args = args.parse_args()
	main(args)