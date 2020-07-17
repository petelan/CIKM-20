# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# load needed libraries 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import sys
from scipy.stats import entropy

# I have 2 source domain and 1 target domain
german = np.genfromtxt('default_degree.csv',delimiter=',')
index_T = np.random.permutation(len(german))
index_kT = index_T[0:2000]
german = german[index_kT]

print(german.shape)

sample = german[:,0:-1] 
label = german[:,-1]

#how many do we choose form unlabeled data
R = 1





#then, we write a data process function
def dataProcess1(trainData):
    #print(whole[:,4])
    trainData1 = np.copy(trainData)
    trainData1[:,0] = (trainData[:,0]-np.mean(trainData[:,0]))/pd.Series(trainData[:,0]).mad()
    trainData1[:,3] = (trainData[:,3]-np.mean(trainData[:,3]))/pd.Series(trainData[:,3]).mad()
    trainData1[:,4] = (trainData[:,4]+2)/10
    trainData1[:,5] = (trainData[:,5]+2)/10
    trainData1[:,6] = (trainData[:,6]+2)/10
    trainData1[:,7] = (trainData[:,7]+2)/10
    trainData1[:,9] = (trainData[:,9]+2)/10
    trainData1[:,10] = (trainData[:,10]+2)/10    
    trainData1[:,11] = (trainData[:,11]-np.mean(trainData[:,11]))/pd.Series(trainData[:,11]).mad()
    trainData1[:,12] = (trainData[:,12]-np.mean(trainData[:,12]))/pd.Series(trainData[:,12]).mad()
    trainData1[:,13] = (trainData[:,13]-np.mean(trainData[:,13]))/pd.Series(trainData[:,13]).mad()
    trainData1[:,14] = (trainData[:,14]-np.mean(trainData[:,14]))/pd.Series(trainData[:,14]).mad()
    trainData1[:,15] = (trainData[:,15]-np.mean(trainData[:,15]))/pd.Series(trainData[:,15]).mad()
    trainData1[:,16] = (trainData[:,16]-np.mean(trainData[:,16]))/pd.Series(trainData[:,16]).mad()
    trainData1[:,17] = (trainData[:,17]-np.mean(trainData[:,17]))/pd.Series(trainData[:,17]).mad()
    trainData1[:,18] = (trainData[:,18]-np.mean(trainData[:,18]))/pd.Series(trainData[:,18]).mad()
    trainData1[:,19] = (trainData[:,19]-np.mean(trainData[:,19]))/pd.Series(trainData[:,19]).mad()
    trainData1[:,20] = (trainData[:,20]-np.mean(trainData[:,20]))/pd.Series(trainData[:,20]).mad()
    trainData1[:,21] = (trainData[:,21]-np.mean(trainData[:,21]))/pd.Series(trainData[:,21]).mad()
    trainData1[:,22] = (trainData[:,22]-np.mean(trainData[:,22]))/pd.Series(trainData[:,22]).mad()

    return trainData1



def randomChoose(L,U):
    index_r = np.random.permutation(len(U))
    index_k1 = index_r[0:R]
    return U[index_k1],index_k1


def bootLogistic(sample_train, label_train):
    nT, p = sample_train.shape
    label_train = label_train.reshape((len(label_train),1))
    for i in range(nT//5):
        index_b = np.random.permutation(len(label_train))
        index_b = index_b[0:20]
        if i == 0:
            sample_train_b = sample_train[index_b]
            label_train_b = label_train[index_b]
        else:
            sample_train_b = np.vstack((sample_train_b, sample_train[index_b]))
            label_train_b = np.vstack((label_train_b, label_train[index_b]))
    
    sample_train_b = np.array(sample_train_b)
    label_train_b = np.array(label_train_b)
    
    classifier = LogisticRegression()
    classifier.fit(sample_train_b,label_train_b)
    

    return classifier


def TwoModel(sample_L,label_L,sample_test,sample_test_p,label_test):
    index_F = np.where(sample_L[:,8] == 0)
    index_M = np.where(sample_L[:,8] == 1)
    sample_TrainF = sample_L[index_F]
    sample_TrainM = sample_L[index_M]
    sample_TrainF = np.delete(sample_TrainF,8,axis = 1)
    sample_TrainM = np.delete(sample_TrainM,8,axis = 1)
    label_LF = label_L[index_F]
    
    label_LM = label_L[index_M]
    
    
    
    index_FT = np.where(sample_test_p[:,8] == 0)
    index_MT = np.where(sample_test_p[:,8] == 1)
    sample_TestFT = sample_test_p[index_FT]
    sample_TestMT = sample_test_p[index_MT]
    sample_TestFT = np.delete(sample_TestFT,8,axis = 1)
    sample_TestMT = np.delete(sample_TestMT,8,axis = 1)
    label_LFT = label_test[index_FT]
    label_LMT = label_test[index_MT]
    
    beta1_All = []
    beta2_All = []
    
    beta1_profile = []
    beta2_profile = []
    
    for i in range(10):
        classifier1 = bootLogistic(sample_TrainF, label_LF)
        label_train_predicted1 = classifier1.predict(sample_TrainF)
        
        label_train_predicted1[np.where(label_train_predicted1<=0.5)] = 0
        label_train_predicted1[label_train_predicted1!=0] = 1
        
        classifier2 = bootLogistic(sample_TrainM, label_LM)
        label_train_predicted2 = classifier2.predict(sample_TrainF)
        label_train_predicted2[np.where(label_train_predicted2<=0.5)] = 0
        label_train_predicted2[label_train_predicted2!=0] = 1
        
        #if sum(label_train_predicted1) not in beta1_profile:
            #beta1_profile.append(sum(label_train_predicted1))
        beta1_All.append(classifier1)
            
        #if sum(label_train_predicted2) not in beta2_profile:
            #beta2_profile.append(sum(label_train_predicted2))
        beta2_All.append(classifier2)
            
    loss = sys.maxsize
    lam = 0.5
    for betaF in beta1_All:
        for betaM in beta2_All:
            loss1 = np.sum(np.abs(label_LF-betaF.predict(sample_TrainF)))
            loss2 = np.sum(np.abs(label_LM-betaM.predict(sample_TrainM)))
            lossl1 = lam*(loss1+loss2)/(len(sample_L))
            loss_Fa = (sum(betaF.predict(sample_TrainF))+sum(betaM.predict(sample_TrainM)))/2
            loss_F = abs(sum(betaF.predict(sample_TrainF)) - loss_Fa) + abs(sum(betaM.predict(sample_TrainM)) - loss_Fa)
            loss_T = lossl1 + (1-lam)*loss_F/(len(sample_L))
            if loss_T < loss:
                loss = loss_T
                betaF1 = betaF
                betaM1 = betaM
        
    predictF = betaF1.predict(sample_TestFT)
    predictM = betaM1.predict(sample_TestMT)
    
    predictF[np.where(predictF<=0.5)] = 0
    predictF[predictF!=0] = 1
    predictM[np.where(predictM<=0.5)] = 0
    predictM[predictM!=0] = 1
    
    
    p1 = np.array(list(predictF) + list(predictM))
    p2 = np.array(list(label_LFT) + list(label_LMT))
    
    MSE = mean_squared_error(p1, p2)
    

    
    PredictP1 = np.sum(predictF == 1)/len(predictF)
    PredictP2 = np.sum(predictM == 1)/len(predictM)   
    PredictP = abs(PredictP1 - PredictP2)
                   
    return betaF1, betaM1, MSE, PredictP, beta1_All, beta2_All
    

#write the FairRank function
def rankFair(L,U,sample_test,sample_test_p,label_test):
    Rank_c = []
    Rank_c3 = []
    sex = L[:,8]
    
    index_FL = np.where(L[:,8] == 0)
    index_ML = np.where(L[:,8] == 1)
    
    sample_sex_out = np.delete(L,8,axis = 1)
    sample_sex = L[:,0:-1]
    label_sex = L[:,-1]
    betaF1, betaM1, MSE, PredictP, beta1_All, beta2_All = TwoModel(sample_sex,label_sex,sample_test,sample_test_p,label_test)
    clf = LogisticRegression().fit(sample_sex_out[:,0:-1], sex)
    
    for i in range(len(U)):
#predict the probability
        Mid = np.delete(U[i,:],8,axis = 0)
        Mid = Mid[0:-1]
        final = betaF1.predict_proba([Mid])[0,0] - betaM1.predict_proba([Mid])[0,0]
        a = list(label_sex[index_FL]).count(1) + list(betaF1.predict([Mid])).count(1)
        b = list(label_sex[index_ML]).count(1) + list(betaM1.predict([Mid])).count(1)
        c = list(label_sex[index_FL]).count(1)
        d = list(label_sex[index_ML]).count(1)
        final3 = abs(clf.predict_proba([Mid])[0,0]*(a/(len(label_sex[index_FL]) + 1) - d/(len(label_sex[index_ML])))) + abs(clf.predict_proba([Mid])[0,1]*(b/(len(label_sex[index_ML])+1) - c/(len(label_sex[index_FL]))))
        Rank_c.append(final)
        Rank_c3.append(final3)
        
        

        
    Rank_c = np.array(Rank_c)
    Rank_c3 = np.array(Rank_c3)
    
    rank_index_c1 = Rank_c.argsort()[0:R]
    rank_index_c2 = (-Rank_c).argsort()[0:R]
    rank_index_c3 = (-Rank_c3).argsort()[0:R]
    
    return U[rank_index_c1],rank_index_c1,U[rank_index_c2],rank_index_c2, U[rank_index_c3],rank_index_c3, MSE, PredictP



sample =  dataProcess1(sample)

n = len(sample)

sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test_p = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]

#delete all the information of sensitive feature?
sample_test = np.delete(sample_test_p,8,axis = 1)

label_train = label_train.reshape((len(label_train),1))
all_train = np.hstack((sample_train,label_train))

k = 100
Times = 10
inter = 500


index_array = []

#for i in range(Times):
#    index_array.append(np.random.permutation(len(sample_train)))
    
#index_save = pd.DataFrame(data = index_array)
#index_save.to_csv('index_save.csv')



index_array = np.genfromtxt('index_save.csv',delimiter=',')
index_array = np.array(index_array)
index_array = index_array.astype(np.int16)


M1_sum = np.zeros(inter)
P1_sum = np.zeros(inter)
P1_na_sum = np.zeros(inter)
PC1_sum = np.zeros(inter)

M2_sum =np.zeros(inter)
P2_sum = np.zeros(inter)
P2_na_sum = np.zeros(inter)
PC2_sum = np.zeros(inter)

M3_sum = np.zeros(inter)
P3_sum = np.zeros(inter)
P3_na_sum = np.zeros(inter)
PC3_sum = np.zeros(inter)

M4_sum =np.zeros(inter)
P4_sum = np.zeros(inter)
P4_na_sum = np.zeros(inter)
PC4_sum = np.zeros(inter)

for x in range(Times):
    index = index_array[x]
    index_k1 = index[0:k]
    index_k2 = index[k:]
    
    sample_L1 = sample_train[index_k1]
    label_L1 = label_train[index_k1]
    sample_L2 = sample_train[index_k1]
    label_L2 = label_train[index_k1]
    
    sample_L3 = sample_train[index_k1]
    label_L3 = label_train[index_k1]
    sample_L4 = sample_train[index_k1]
    label_L4 = label_train[index_k1]
    
    # All the inforamtion in L and U
    L1 = all_train[index_k1]
    U1 = all_train[index_k2]
    
    L2 = all_train[index_k1]
    U2 = all_train[index_k2]

    L3 = all_train[index_k1]
    U3 = all_train[index_k2]
    
    L4 = all_train[index_k1]
    U4 = all_train[index_k2]
    
    M1 = []
    P1 = []
    P1_na = []
    PC1 = []

    M2 = []
    P2 = []
    P2_na = []
    PC2 = []

    M3 = []
    P3 = []
    P3_na = []
    PC3 = []

    M4 = []
    P4 = []
    P4_na = []
    PC4 = []


    #Twomodel
    for p in range(inter):
        High_rank_c1,index_c1,High_rank_c2,index_c2, High_rank_c3,index_c3, MSE, PredictP  = rankFair(L3,U3,sample_test,sample_test_p,label_test)
        M3.append(MSE)
        P3.append(PredictP)
        #print(High_rank_c2[0][8],High_rank_c2[0][-1])
        L3 = np.row_stack((L3, High_rank_c2))
        U3 = np.delete(U3,index_c2,axis = 0)
        sample_L3 = L3[:,0:-1]
        label_L3 = L3[:,-1]

    M3_sum = np.add(M3_sum, M3)
    P3_sum = np.add(P3_sum, P3)

    #entropy of result
    for pp in range(inter):
        High_rank_c1,index_c1,High_rank_c2,index_c2, High_rank_c3,index_c3, MSE, PredictP  = rankFair(L1,U1,sample_test,sample_test_p,label_test)
        M1.append(MSE)
        P1.append(PredictP)
        #print(High_rank_c2[0][8],High_rank_c2[0][-1])
        L1 = np.row_stack((L1, High_rank_c3))
        U1 = np.delete(U1,index_c3,axis = 0)
        sample_L1 = L1[:,0:-1]
        label_L1 = L1[:,-1]

    M1_sum = np.add(M1_sum, M1)
    P1_sum = np.add(P1_sum, P1)

    
#random choose        
    for n in range(inter):
        random, index_r = randomChoose(L2,U2)
        betaF1, betaM1, MSE, PredictP, beta1_All, beta2_All = TwoModel(sample_L2,label_L2,sample_test,sample_test_p,label_test)
        M2.append(MSE)
        P2.append(PredictP)
        L2 = np.row_stack((L2, random))
        U2 = np.delete(U2,index_r,axis = 0)
        sample_L2 = L2[:,0:-1]
        label_L2 = L2[:,-1]        
    M2_sum = np.add(M2_sum, M2)
    P2_sum = np.add(P2_sum, P2)



#plot the result
plt.xlabel('Times of queried demographics')
plt.ylabel('Classification Error')    
plt.plot(np.array(M2_sum)/Times, label='Classification Error of random choose')
plt.plot(np.array(M3_sum)/Times, label='Classification Error of QmC')
plt.plot(np.array(M1_sum)/Times, label='Classification Error of QmCs')
plt.legend()
plt.show()    

plt.xlabel('Times of queried demographics')
plt.ylabel('Statistical Parity')   
plt.plot(np.array(P2_sum)/Times, label='Statistical Parity of random choose')
plt.plot(np.array(P3_sum)/Times, label='Statistical Parity of QmC')
plt.plot(np.array(P1_sum)/Times, label='Statistical Parity of QmCs')
plt.legend()
plt.show()



#save the result
M1_sum = pd.DataFrame(data = np.array(M1_sum)/Times)
M1_sum.to_csv('M1_sum.csv')
M2_sum = pd.DataFrame(data = np.array(M2_sum)/Times)
M2_sum.to_csv('M2_sum.csv')
M3_sum = pd.DataFrame(data = np.array(M3_sum)/Times)
M3_sum.to_csv('M3_sum.csv')


P1_sum = pd.DataFrame(data = np.array(P1_sum)/Times)
P1_sum.to_csv('P1_sum.csv')
P2_sum = pd.DataFrame(data = np.array(P2_sum)/Times)
P2_sum.to_csv('P2_sum.csv')
P3_sum = pd.DataFrame(data = np.array(P3_sum)/Times)
P3_sum.to_csv('P3_sum.csv')





