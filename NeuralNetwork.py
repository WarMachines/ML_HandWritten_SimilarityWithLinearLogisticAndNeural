import numpy as np               # numpy library being imported to use in the code to use arrays objects, scientific calculation related with linear algebra,permtation, randomization and many more
import tensorflow as tf          # tensorflow is an open source library used for numerical computational like matrix multiplication neural network related operation
from tqdm import tqdm_notebook   # It was used to show progress bar related information
import pandas as pd              # Pandas library provides flexible data structures for complex objects     
from keras.utils import np_utils # Utilities provided by keras
import math
import matplotlib.pyplot as plt

# Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

def encodeTargets(targets):
    processedLabel = []
    for target in targets:
        if target == 1:
            processedLabel.append([1])
        else:
            processedLabel.append([0])
    return  np_utils.to_categorical(np.array(processedLabel),2)

def NeuralNetwork(InputSet,InputTargetSet,InputDescriptionSet):
    NUM_HIDDEN_NEURONS_LAYER_1 = 500                   # defining the number of layer we have at hidden layer or neuron layers
    LEARNING_RATE = 0.01                          # Learning rate for our model, it define the freqency of the changes in parameters
                                                    # Lower the learning rate slows learning process but tends to produce high accuracy
    for test in range(0,len(InputSet)):
        print(InputDescriptionSet[test])
        featureSet = InputSet[test]
        targetSet = np.transpose(np.mat(InputTargetSet[math.floor(test/2)]))
        #Separation length for training and testing
        trainingLength = math.ceil(len(featureSet) * 0.8)

        # Create Training Data sets
        processedTrainingData = (featureSet)[slice(trainingLength)]
        processedTrainingLabel= encodeTargets((targetSet)[slice(trainingLength)])

        #Create Testing DataSets
        processedTestingData = (featureSet)[slice(trainingLength+1,len(featureSet))]
        processedTestingLabel = (targetSet)[slice(trainingLength+1,len(featureSet))]
        
        inputTensor  = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
        outputTensor = tf.placeholder(tf.float32, [None, 2])
        input_hidden_weights  = init_weights([featureSet.shape[1], NUM_HIDDEN_NEURONS_LAYER_1])
        hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 2])

        hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))

        output_layer = tf.matmul(hidden_layer, hidden_output_weights)

        error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=outputTensor))

        training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

        prediction = tf.argmax(output_layer, 1)

        NUM_OF_EPOCHS = 500
        BATCH_SIZE = 64

        training_accuracy = []

        with tf.Session() as sess:  # Tensorflow session starts here
            
            tf.global_variables_initializer().run()
            
            for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
                #Shuffle the Training Dataset at each epoch
                p = np.random.permutation(range(len(processedTrainingData)))
                processedTrainingData  = processedTrainingData[p]
                processedTrainingLabel = processedTrainingLabel[p]
                # Start batch training
                for start in range(0, len(processedTrainingData), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                                outputTensor: processedTrainingLabel[start:end]})
                # Training accuracy for an epoch
                training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                                    sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                                    outputTensor: processedTrainingLabel})))
            # Testing    (Testing on testing data 1-100)
            predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})

            wrong   = 0
            right   = 0

            predictedTestLabelList = []

            for i,j in zip(processedTestingLabel,predictedTestLabel):
                
                if np.argmax(i) == j:  
                    right = right + 1 # If predicted value is right, increment variable right
                else:
                    wrong = wrong + 1 # If predicted value is wrong, increment variable wrong

            print("Errors: " + str(wrong), " Correct :" + str(right))
            print("Testing Accuracy: " + str(right/(right+wrong)*100))  # Checking the Accuracy of the output by formula

            #Accuracy Graph between epoch and training_accuracy
            plt.xlim(1, NUM_OF_EPOCHS)
            plt.plot(training_accuracy)
            plt.xlabel("epoch")
            plt.ylabel("Training Accuracy Values")
            plt.savefig("Neural Network Training Accuracy vs epochs "+ InputDescriptionSet[test])
            plt.close()