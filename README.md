Training a model through Behavioral Cloning!

Training network was based on Dave-2 "https://developer.nvidia.com/blog/deep-learning-self-driving-cars/"    - (paper: https://gvelarde.com/Press/Teaching/G2_Bruckner_Hervas_Martinez_Ugarte_Driving.pdf)

Training sim was used from: https://github.com/udacity/self-driving-car-sim






-----STEPS-----

1.Generate data (rgb pictures) for the environment using manual drive

2.Create a model and train it with the collected data - **model.py**

3.Use your model to drive the car and see the results - **drive.py**

(an h5 object will be generated with the model weights. in order for the model to be used, you must first be on Autonomous Mode and call the drive.py file with the equivalent weight file eg. "cd path " - " python drive.py model_1.h5")

---------------







performance example in https://www.youtube.com/watch?v=HNtmUKSBbp4
