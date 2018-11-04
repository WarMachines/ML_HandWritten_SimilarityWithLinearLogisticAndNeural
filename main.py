from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import pandas as pd
import LinearRegression as linearReg
import LogisticRegression as logisReg
import NeuralNetwork as neuralNet


human_observe_feature_dict = {}
human_observe_feature_subtractSet = []
human_observe_feature_concatenateSet = []
human_observe_feature_targetVector = []
gsc_feature_dict = {}
gsc_feature_subtractSet = []
gsc_feature_concatenateSet = []
gsc_feature_targetVector = []

total_input_matrix_sets = []
total_target_vector_sets = []
total_input_matrix_sets_description = ["Human Observed Dataset with feature subtraction","Human Observed Dataset with feature concatentation","GSC Dataset with feature subtraction","GSC Dataset with feature concatentation"]

def FormatFeatureSet(filePath):
    featureSet = pd.read_csv(filePath,index_col='img_id') 
    featureSet.drop(featureSet.columns[0], axis=1,inplace=True)
    feature_dict = {}
    for index, row in featureSet.iterrows():
        feature_value_list = []
        for feature in range(0,len(row)):
            feature_value_list.append(row[feature])
        feature_dict[index] = feature_value_list

    return feature_dict

def BuildFeatureSetsAll(samePairSetPath, diffPairSetPath, featureSetDict):
    samePairSet = pd.read_csv(samePairSetPath)
    diffPairSet = pd.read_csv(diffPairSetPath)
    diffPairSet = diffPairSet.sample(n=len(samePairSet))
    totalPairSet = samePairSet.append(diffPairSet).sample(frac=1)

    substractDataMatrix = []
    concatenateDataMatrix = []
    targetVector = [];
    count = 0;
    for index, row in totalPairSet.iterrows():
            substractDataMatrix.append(np.absolute(np.subtract(featureSetDict[row[0]],featureSetDict[row[1]])))
            concatenateDataMatrix.append(np.concatenate((featureSetDict[row[0]],featureSetDict[row[1]]),axis=None))
            targetVector.append(row[2])
    return ValidateAndRemoveColumns(substractDataMatrix),ValidateAndRemoveColumns(concatenateDataMatrix),targetVector

def ValidateAndRemoveColumns(featureSet):
    variances = []
    columnIndexToRemove = []
    variances = np.var(featureSet,axis=0)
    for index in range(0,len(variances)):
        if variances[index] == 0:
            columnIndexToRemove.append(index)
    return np.delete(featureSet, columnIndexToRemove, axis=1)        


human_observe_feature_dict = FormatFeatureSet("given\\HumanObserved-Dataset\\HumanObserved-Features-Data\\HumanObserved-Features-Data.csv")
human_observe_feature_subtractSet,human_observe_feature_concatenateSet,human_observe_feature_targetVector = BuildFeatureSetsAll("given\\HumanObserved-Dataset\\HumanObserved-Features-Data\\same_pairs.csv","given\\HumanObserved-Dataset\\HumanObserved-Features-Data\\diffn_pairs.csv",human_observe_feature_dict)
print (len(human_observe_feature_subtractSet),len(human_observe_feature_concatenateSet))
print (len(human_observe_feature_subtractSet[0]),len(human_observe_feature_concatenateSet[0]))

# gsc_feature_dict = FormatFeatureSet("given\\GSC-Dataset\\GSC-Features-Data\\GSC-Features.csv")
# gsc_feature_subtractSet,gsc_feature_concatenateSet,gsc_feature_targetVector = BuildFeatureSetsAll("given\\GSC-Dataset\\GSC-Features-Data\\same_pairs.csv","given\\GSC-Dataset\\GSC-Features-Data\\diffn_pairs.csv",gsc_feature_dict)
# print (len(gsc_feature_subtractSet),len(gsc_feature_concatenateSet))
# print (len(gsc_feature_subtractSet[0]),len(gsc_feature_concatenateSet[0]))

total_input_matrix_sets.append(human_observe_feature_subtractSet)
total_input_matrix_sets.append(human_observe_feature_concatenateSet)
total_target_vector_sets.append(human_observe_feature_targetVector)
# total_input_matrix_sets.append(gsc_feature_subtractSet)
# total_input_matrix_sets.append(gsc_feature_concatenateSet)
# total_target_vector_sets.append(gsc_feature_targetVector)


print ('UBITname      = chauhan9')
print ('Person Number = 50290975')
print ('Name          = Saurabh kumar Chauhan')


# Linear regression solution with close form and Gradient descent
linearReg.LinearRegressionSolution(total_input_matrix_sets,total_target_vector_sets,total_input_matrix_sets_description)

#Logistic Regression solution with gradient descent using sigmoid function
logisReg.logisticRegression(total_input_matrix_sets,total_target_vector_sets,total_input_matrix_sets_description)

#Neural network training
neuralNet.NeuralNetwork(total_input_matrix_sets,total_target_vector_sets,total_input_matrix_sets_description)