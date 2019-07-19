#An image recognition system designed to recognize images of bees and images of the number 3
#My little tribute to CGP Grey
#Powered by ImageAI

from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("Bor3")
model_trainer.trainModel(num_objects=2, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)
