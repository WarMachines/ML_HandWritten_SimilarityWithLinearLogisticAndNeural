
from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import pandas as pd


 #function definitions for the linear regression codes
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):            # Slicing the training target set from entire target dataset (80% of the entire sample set)
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):           # slicing 80% of the entire sample as training data
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount):                # Generating validation data set (10% of the data)
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount):        # Generating target vector for validation data set
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

# Calculate big sigma to find the design matrix, It will be same for entire model, means it wont change for any data set's design matrix calculation
def GenerateBigSigma(Data, MuMatrix,TrainingPercent):
    BigSigma    = np.zeros((len(Data),len(Data)))       # BigSigma of shape 41*41, where data(shape 41*total sample) is list of row = feature values, column = sample values
    DataT       = np.transpose(Data)                    # now DataT(shape of total sample * 41)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        # reducing to iterate over 80% data from sample
    varVect     = []
    for i in range(0,len(DataT[0])):                #will run 41 times (to loop features)
        vct = []
        for j in range(0,int(TrainingLen)):         # will run 80% of total sample data
            vct.append(Data[i][j])                  # extract feature value of each sample
        varVect.append(np.var(vct))                 # find the variance of that feature over each sample(80% of total)
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]                 # we have 41 variance for each feature set them in diagonal of big sigma matrix
   
    BigSigma = np.dot(200,BigSigma)             # multiple the matrix to scaler to increase the significance of numbers
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)            # 1*41 size matrix  - 1*41 size matrix
    T = np.dot(BigSigInv,np.transpose(R))     # 41*41 bigSigmaInv matrix dot 41*1 matrix(means one feature array)
    L = np.dot(R,T)                           # 1*41 matrix dot 41*1 matrix    (This will be a scalar value)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))     # calculate entries for design matrix
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)                                  # DataT size samples*41
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))  # 80% of the samples   
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))            # phi matrix or design matrix of size 80% of sample* 10 (as we have 10 cetroids from clusters)
    BigSigInv = np.linalg.inv(BigSigma)                         # (inverse of the variance matrix)to find inverse matrix should be a squared matrix which 41*41 for now   
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)       # find the basis functions for design matrix (80% of samples* 10)
    #print ("PHI Generated..")
    return PHI  

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))    # 10*1 dot 10* 80% of sample
    ##print ("Test Out Generated..")
    return Y

# Calculate the ERMS values between the difference of actual target values and predicted target value by model
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda                      # 10*10 Least square regulaization l with value 0.03 diagonally 
    PHI_T       = np.transpose(PHI)                  # 10 * 80% of sample   transpose of the design matrix
    PHI_SQR     = np.dot(PHI_T,PHI)                  # 10 * 10  dot product of design matrix and its transpose
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)           # 10*10 - 10*10 matrix   Add regulizer
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)          # 10*10  Inverse of the total resultant matrix
    INTER       = np.dot(PHI_SQR_INV, PHI_T)         # 10* 80% of sample  dot product if the result matrix with transpose of design matrix
    W           = np.dot(INTER, T)                   # (10*80% of sample) * (80% of sample target * 1)   dot product with the target matrix to get weight matrix
    ##print ("Training Weights Generated..")
    return W 


def LinearRegressionSolution(InputTestSet,InputTestTargetSet,InputTestDescriptionSet):
    #variable declaration
    maxAcc = 0.0
    maxIter = 0
    C_Lambda = 0.03
    TrainingPercent = 80
    ValidationPercent = 10
    TestPercent = 10
    M = 10
    PHI = []

    for test in range(0,len(InputTestSet)):
        print("*********************************************************************************************")
        print(InputTestDescriptionSet[test])
        print("*********************************************************************************************")
        RawTarget = InputTestTargetSet[math.floor(test/2)]
        RawData =np.transpose(InputTestSet[test])

    # Preparing training Data
        TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
        TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
        print(TrainingTarget.shape)
        print(TrainingData.shape)

    # Preparing validation Data
        ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
        ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
        print(ValDataAct.shape)
        print(ValData.shape)

    # Preparing testing Data
        TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
        TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
        print(ValDataAct.shape)
        print(ValData.shape)

    #Close form solution
        print("_________________Close Form Solution________________________")
        ErmsArr = []
        AccuracyArr = []

        kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
        Mu = kmeans.cluster_centers_

        BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent)    # 41 * 41 find the sigma matrix of variance(diagonal) find the variance of a type feature values and put them diagonally 
        TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)          # 80% of sample * 10   design matrix for training set
        W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))  # 10 elements of a vector find the weight matrix
        TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100)                     # 10% of sample * 10
        VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)                      # 10% of sample * 10
        print( "#############################")
        print(Mu.shape)
        print(BigSigma.shape)
        print(TRAINING_PHI.shape)
        print(W.shape)
        print(VAL_PHI.shape)
        print(TEST_PHI.shape)

        print ("************************")
        TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
        VAL_TEST_OUT = GetValTest(VAL_PHI,W)
        TEST_OUT     = GetValTest(TEST_PHI,W)

        TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))    # get root mean squared values (root(sum(1-n) of square(predictedTarget - expectedTarget))
        ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))       # get root mean squared values (root(sum(1-n) of square(predictedTarget - expectedTarget))
        TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))          # get root mean squared values (root(sum(1-n) of square(predictedTarget - expectedTarget))
        print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
        print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
        print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))


    #  gradient descent solution
        print("_________________Gradient Descent Solution________________________")
        W_Now        = np.dot(1, W)        # initialize weight randomly (here we working with the weight matrix generated from close form solution)
        La           = 2                     # Lambda value for regularizer
        learningRate = 0.01                  # Learning rate for the model
        L_Erms_Val   = []                    
        L_Erms_TR    = []
        L_Erms_Test  = []
        W_Mat        = []

        for i in range(0,400):                         # 400 iteration, iterative weight updation for finding the optimal values
            
            #print ('---------Iteration: ' + str(i) + '--------------')
            Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])    # Find the Delta_E_D
            La_Delta_E_W  = np.dot(La,W_Now)                                              # Find the La_Delta_E_W
            Delta_E       = np.add(Delta_E_D,La_Delta_E_W)                                # Find the Delta_E    
            Delta_W       = -np.dot(learningRate,Delta_E)                                 # Find the Delta_W
            W_T_Next      = W_Now + Delta_W                                               # Find the next weight vector by addign the gradient of weight 
            W_Now         = W_T_Next                                                      # Set the current weight to calculated weight for next iteration
            
            #print(W_Now)
            #-----------------TrainingData Accuracy---------------------#
            TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
            Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))
            
            #-----------------ValidationData Accuracy---------------------#
            VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
            Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))
            
            #-----------------TestingData Accuracy---------------------#
            TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
            Erms_Test = GetErms(TEST_OUT,TestDataAct)
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
        print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
        print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))

        plt.xlim(1, 400)
        plt.plot(L_Erms_TR)
        plt.xlabel("iteration")
        plt.ylabel("Training ERMS Values")
        plt.savefig("Linear Regression Training ERMS vs Iterations for "+InputTestDescriptionSet[test])
        plt.close()

        plt.xlim(1, 400)
        plt.plot(L_Erms_Val)
        plt.xlabel("iteration")
        plt.ylabel("Validation ERMS Values")
        plt.savefig("Linear Regression Validation ERMS vs Iterations for "+InputTestDescriptionSet[test])
        plt.close()

        plt.xlim(1, 400)
        plt.plot(L_Erms_Test)
        plt.xlabel("iteration")
        plt.ylabel("Testing ERMS Values")
        plt.savefig("Linear Regression Testing ERMS vs Iterations for "+InputTestDescriptionSet[test])
        plt.close()