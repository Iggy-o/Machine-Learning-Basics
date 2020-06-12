'''
Ighoise Odigie
June 11, 2020
Youtube: https://www.youtube.com/channel/UCud4cJjtCjEwKpynPz-nNaQ?
Github: https://github.com/Iggy-o
Preview: https://repl.it/@IghoiseO/Machine-Learning-Basics#main.py
'''

#MACHINE LEARNING: EXPLAINED
'''
  I'm relatively new to the world of machine learning but here's what I've
  learned. Originally, the AI is initialized with completely new brain made 
  up of random connections and as a result like a baby it is very ineffective. 
  But it is trained against a premade data set and its connections are tweaked 
  to make it's answers closer to the dataset. Hopefully after training the AI's 
  brain can answer related questions accurately.
  The AI's brain structure resembles a tree =>    o     Result
                                                / | \   Weights/Connections
                                               o  o  o  Inputs
'''

#Here is an amazing tutorial series
'''
https://www.youtube.com/playlist?list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
'''

#The Instructions
instructions = '''
---BASICS OF MACHINE LEARNING---


How To Use
----------
The Test Data is currently set up for Phones Vs. Laptops. 
The program should learn from the premade data and answer 
your given input correctly, more iterations results in 
better answers. You must input in the following format:
-----------
Screen Size(Inches)
Battery Life(Hrs)
Weight(lb)
# of AI Interations(a.k.a Wait Time)
-----------
\n\n
               '''

#A few libraries must be imported for the program to run
import numpy as np

#The instructions are displayed to the user and they are prompted to enter values for the mystery node below
print(instructions)
screenSize = float(input("What is your device's screen size => "))
batteryLife = float(input("What is your device's battery life => "))
weight = float(input("What is your device's weight=> "))

#The training rate defines the size of the change between iterations, the attempts refers to the number of iterations
training_Attempts = int(input("How many iterations do you want => "))
training_Rate = 0.3


#<!--The Training Data-->

#This variable stores the data which the AI is trained against (Screen Size, Weight[kg], Answer)
data = [
        [19, 7, 20, 1],
        [20, 6, 21, 1],
        [21, 7, 21, 1],
        [22, 8, 22, 1],
        [23, 4, 25, 1],
        [24, 5, 23, 1],
        [25, 9, 26, 1],
        [26, 5, 24, 1],
        [27, 8, 27, 1],
        [28, 6, 28, 1],
        [29, 7, 30, 1],
        [30, 7, 30, 1],
        [31, 7, 29, 1],
        [32, 6, 24, 1],
        [33, 9, 21, 1],
        [4, 17, 0.3, 0],
        [7, 8, 0.3, 0],
        [5, 16, 0.4, 0],
        [6, 15, 0.45, 0],
        [6, 12, 0.5, 0],
        [7, 17, 0.8, 0],
        [6, 14, 0.6, 0],
        [4, 13, 0.4, 0],
        [6, 10, 0.6, 0],
        [5, 9, 0.6, 0],
        [6, 16, 0.45, 0],
        [7, 17, 0.5, 0],
        [9, 17, 0.55, 0],
        ]

#The mystery node is the final test to see if the AI is properly trained
mystery_node = [screenSize, batteryLife, weight]

#<!--The Artificial Brain-->

#There are four weights and a bias (a.k.a The Brain's connections) that are assigned randomly and altered over several iterations to get a more accurate answer
initValue = np.random.randn()
weight1 = initValue
weight2 = initValue
weight3 = initValue
bias = initValue

#This class holds most of the machine learning algorithm
class machineBrain:
  #When you call this class inputs must be provided
  def __init__(self, a, b, c, d = "null"):
    self.input1 = a
    self.input2 = b
    self.input3 = c
    self.target = d

  #Sigmoid is just an "S" shaped squashing function that squashes a very large or small value between 1 and 0
  def sigmoid(self, value):
    return 1/(1 + np.exp(-value))
  def d_sigmoid(self, value):
    return self.sigmoid(value) * (1 - self.sigmoid(value))

  #This function returns the computer's prediction on a 0 to 1 scale
  def prediction(self):
    prediction = weight1 * self.input1 + weight2 * self.input2 + weight3 * self.input3 + bias
    squashedPrediction = self.sigmoid(prediction)
    return prediction, squashedPrediction
  
  #This function calculates how far from correct the computer's answer is
  def err(self):
    pred, squashPred = self.prediction()
    err = np.square(squashPred - self.target)
    return err

  #This function attempts to make the computer's guesses closer to the answer's given by the data
  #This is done by calculating the derivative back to the weights, which a graph can be derived from
  #The weights are altered based on the slopes of the derivative's graph
  #This can push the computer's error to the lowest possible value and make them closer to the target answer
  #Hopefully that point is 0 and after repeating this error suppression process
  #The computer will hopefully be pretty accurate
  def errSuppress(self):
    #Create a bunch of derivitaves for each function that is used
    pred, squashPred = self.prediction()
    derr_dSquashpred = 2 * (squashPred - self.target)
    dSquashpred_dpred = self.d_sigmoid(pred)
    dpred_dw1 = self.input1
    dpred_dw2 = self.input2
    dpred_dw3 = self.input3
    dpred_db = 1

    #Combine the derivatives from above into general ones for each weight (and bias)
    derr_dpred = derr_dSquashpred * dSquashpred_dpred
    derr_dw1 = derr_dpred * dpred_dw1
    derr_dw2 = derr_dpred * dpred_dw2
    derr_dw3 = derr_dpred * dpred_dw3
    derr_db = derr_dpred * dpred_db

    #Finally the weights are altered based on the training rate
    global weight1, weight2, weight3, bias
    weight1 -= derr_dw1*training_Rate
    weight2 -= derr_dw2*training_Rate
    weight3 -= derr_dw3*training_Rate
    bias -= derr_db*training_Rate

    #Returns the margin of error
    return self.err()

#The following loop is for the training process of the AI
print("Training...")
for attempts in range(training_Attempts):
  for data_point in range(len(data)):
    brain = machineBrain(data[data_point][0], data[data_point][1], data[data_point][2], data[data_point][3])
    error = brain.errSuppress()

#After the AI is trained it will output based on the user's mystery node
print("Complete") 
brain = machineBrain(mystery_node[0], mystery_node[1],mystery_node[2])
pred, Squashpred = brain.prediction()
if Squashpred > 0.5:
  answer = "Your device is a Laptop"
else:
  answer = "Your device is a Phone"
print(f"\nThe computer says: {answer}")
#print(f"The weights: {weight1}, {weight2}, {weight3}")