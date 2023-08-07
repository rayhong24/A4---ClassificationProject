# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
#

# Perceptron implementation
import util
import heapq

PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                #compute score for each label:
                scores = {}
                for label in self.legalLabels:
                    scores[label] = trainingData[i]*self.weights[label]
                    # score = 0
                    # for feature in self.features:
                    #     score += trainingData[i][feature]*self.weights[label][feature]
                    # scores[label] = score

                #find the most optimum label:
                guess_label = max(scores, key=scores.get)

                #update weight if necessary:
                actual_label = trainingLabels[i]
                if guess_label != actual_label:
                    self.weights[actual_label] += trainingData[i]
                    self.weights[guess_label] -= trainingData[i]
                    # for feature in self.features:
                    #     self.weights[actual_label][feature] += trainingData[i][feature]
                    #     self.weights[guess_label][feature] -= trainingData[i][feature]

        print("finished training")



    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        num_in_q = 0
        q = []

        "*** YOUR CODE HERE ***"
        for feature in self.weights[label]:
            if num_in_q >= 100:
                heapq.heappushpop(q, (self.weights[label][feature], feature))
            else:
                heapq.heappush(q, (self.weights[label][feature], feature))
                num_in_q += 1
        
        for w, f in q:
            featuresWeights.append(f)

        return featuresWeights
