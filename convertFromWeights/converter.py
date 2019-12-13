# converting from keras file to tflite file using tflite converte
import tensorflow.lite as lite



def convert(filename):
	converter = lite.TFLiteConverter.from_keras_model_file(filename)
	tflite_model = converter.convert()
	open('../weight_files/foodid_graph.lite','wb').write(tflite_model) 

def main():
	convert('../weight_files/weights-50-0.87.hdf5')


if __name__ == '__main__':
	main()