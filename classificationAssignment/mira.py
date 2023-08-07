# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
import heapq
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        bestAccuracyCount = -1  # best accuracy so far on validation set
        # cGrid.sort(reverse=True)
        bestParams = cGrid[0]
        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING
        for c in cGrid:
            accuracy, weights = self.get_c_accuracy(trainingData, trainingLabels, validationData, validationLabels, c)
            if accuracy > bestAccuracyCount:
                bestAccuracyCount = accuracy
                self.weights = weights

        print("finished training. Best cGrid param = ", bestParams)

    def get_c_accuracy(self, trainingData, trainingLabels, validationData, validationLabels, c):
        weights = self.weights.copy()

        max_accuracy = -1
        max_weights = weights
        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                f = trainingData[i]
                scores = []
                for label in self.legalLabels:
                    scores.append((label, trainingData[i]*weights[label]))

                #find the most optimum label:
                guess_label = max(scores[::-1], key=lambda x:x[1])[0]

                #calculate new weight if necessary:
                actual_label = trainingLabels[i]
                if guess_label != actual_label:
                    diff = weights[guess_label]-weights[actual_label]
                    numerator = diff*f+1
                    denominator = f*f*2

                    tau = min(c, numerator/denominator)

                    f_times_tau = f.copy()
                    f_times_tau.multiplyAll(tau)

                    weights[actual_label] += f_times_tau
                    weights[guess_label] -= f_times_tau

            #calculate accuracy
            correct_count = 0 
            for i in range(len(validationData)):
                f = validationData[i]
                actual_label = validationLabels[i]

                best_score, guess_label = max([(f*weights[label], label) for label in self.legalLabels])

                if guess_label == actual_label:
                    correct_count += 1
                
            accuracy = correct_count / len(validationData)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_weights = weights.copy()
            print("iteration", iteration, "c =", c, "accuracy =", accuracy)

        return max_accuracy, max_weights


    def classify(self, data):
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

        for feature in self.weights[label]:
            if num_in_q >= 100:
                heapq.heappushpop(q, (self.weights[label][feature], feature))
            else:
                heapq.heappush(q, (self.weights[label][feature], feature))
                num_in_q += 1
        
        for w, f in q:
            featuresWeights.append(f)

        return featuresWeights