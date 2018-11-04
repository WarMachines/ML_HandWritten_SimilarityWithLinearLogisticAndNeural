import numpy as np
import matplotlib.pyplot as plt
import math


iteration = 1000
learningRate = 0.01
cost_timeline = []
TrainingRatio = 0.8

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
def Gradient_Descent(x,y):
    w_now = np.mat(np.ones((x.shape[1],1)))
    x= np.mat(x)
    y = np.mat(y)
    for i in range(iteration):
        sigmoid = Sigmoid(np.dot(x,w_now))
        w_now = w_now - learningRate * (np.dot(np.transpose(x),sigmoid - np.transpose(y))/x.shape[1])
        cost_timeline.append(costReg(w_now,x,y,learningRate,sigmoid))
    return w_now

def Classsifier(x,w):
    prob = Sigmoid(x * w)
    return [1 if x >= 0.5 else 0 for x in prob], prob

def costReg(weights, X, y, learningRate, sigmoid):
    weights = np.matrix(weights)
    X = np.matrix(X)
    y = np.matrix(y)  
    first = np.dot(-y, np.log(sigmoid))
    flag = (1-sigmoid).all()
    if flag == False:
        sigmoid-=1e-5
    second = np.dot((1 - y), np.log(1 -sigmoid))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(weights[:,1:weights.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

def plot_decision_boundary(prediction, Description):
    trues = []
    falses = []
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            trues.append(prediction[i])
        else:
            falses.append(prediction[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    no_of_preds = len(trues) + len(falses)
    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')
    plt.legend(loc='upper right');
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.ylim(0, 1)
    plt.savefig("Logistic Regression scatter Graph" + Description)
    plt.close()

def logisticRegression(InputSet,InputTargetSet,InputDescriptionSet):
    for test in range(0,len(InputSet)):
        print(InputDescriptionSet[test])
        #Select current set to work with
        featureSet = InputSet[test]
        targetSet = InputTargetSet[math.floor(test/2)]

        # create Bias Vector for to add to featureSet
        bias = np.ones((featureSet.shape[0],1))    

        #Add bias vector to featureSet
        featureSet = np.hstack((bias,featureSet))

        #Separation length for training and testing
        trainingLength = math.ceil(len(featureSet) * TrainingRatio)

        # Create Training Data sets
        trainingFeatureSet = (featureSet)[slice(trainingLength)]
        trainingTargetSet= (targetSet)[slice(trainingLength)]

        #Create Testing DataSets
        TestingFeatureSet = (featureSet)[slice(trainingLength+1,len(featureSet))]
        TestingTargetSet = (targetSet)[slice(trainingLength+1,len(featureSet))]

        #Calculate Weights using Gradient_Descent
        weights = Gradient_Descent(trainingFeatureSet,trainingTargetSet)

        #Predict the result based on the trained weights
        predictions,predictedProbabilities = Classsifier(TestingFeatureSet,weights)

        #Find the correctness of the predicted targets with actual targets
        correctPredictions = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, TestingTargetSet)]

        # Calculate the accuracy of the model
        accuracy = (sum(correctPredictions) / len(correctPredictions)) * 100
        print ('accuracy = {0}%'.format(np.around(accuracy,2)))

        #Plot the scattered graph of the predicted probabilities
        plot_decision_boundary(predictedProbabilities, InputDescriptionSet[test])

        plt.xlim(1, iteration)
        plt.plot(cost_timeline)
        plt.xlabel("iteration")
        plt.ylabel("Cost Values")
        plt.savefig("Logistic Regression Cost vs iteration "+ InputDescriptionSet[test])
        plt.close()
