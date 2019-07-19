#!/usr/bin/python
#An image recognition system designed to recognize images of bees and images of the number 3
#My little tribute to CGP Grey
#Powered by ImageAI

#get the stuff needed to run prediction ith ImageAI's format
from imageai.Prediction.Custom import CustomImagePrediction
import os, sys

#ensure the user has input one and only one command line argument
#if so, assign the variable inputimage the value of the argument
if len(sys.argv) == 2:
	inputimage=sys.argv[1]
else:
	sys.exit(1)

#used to clear the screen of unnecessary tensorflow output, handling multiple OSes
def clear():
    #windows 
    if os.name == 'nt': 
        _ = os.system('cls') 
    #mac/linux
    else: 
        _ = os.system('clear') 

execution_path = os.getcwd()

#ImageAI prediction with custom ResNet model, trained on 4000 images each of bees/threes
predictor = CustomImagePrediction()
predictor.setModelPath(model_path=os.path.join(execution_path, "model_ex-200_acc-1.000000.h5"))
predictor.setJsonPath(model_json=os.path.join(execution_path, "model_class.json"))
predictor.setModelTypeAsResNet()
predictor.loadModel(num_objects=2)

results, probabilities = predictor.predictImage(image_input=os.path.join(execution_path, inputimage), result_count=2)

#clear the screen, then print the predictor results in a pretty format
clear()
print("Predictor Results: ")
for eachPrediction, eachProbability in zip(results, probabilities):
    print(eachPrediction , " : " , eachProbability)