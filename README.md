# NeuralNetworkApplications
Implementation of neural networks and its application on digit classifer. Also has an application for digit classifier to solve a sudoku given an image.

NeuralNetwork.py - core implementation of stochastic gradient descent back propogation neural network.
Counter.py - First apllication on neural network that learns to count binary sequence.
DigitClassifier.py - reads a csv file that has pixel data of numbers, trains a neural network and stores it as a .pkl file.
SudokuSolver.py - reads a image of a sudoku and solves the puzzle by loading a trained network from .pkl file.

Additional files:
sudo.csv - trainig pixel data for a single font of number 1 to 9 and blank space. 
sudoku.png - image that Solver reads and solves.
train.csv - MINST data set for training. This need to be trained for hand written square sudokus.
test.csv - MINST data for kaggle submission that got 0.97 accuracy on above neural network.
sample_submission.csv - ignore this file. From Kaggle.
trainsub.csv - a subset of traing data for initial trial on small subset.
first.pkl - weights of trained neural network that sudoku solver uses.

Additional libraries used:
Numpy.
PIL - python image library


