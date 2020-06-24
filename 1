import csv
import random
import math

def loadcsv(iris):
    lines=csv.reader(open(iris,"r"))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset

def splitdataset(dataset,splitratio):
    trainsize=int(len(dataset)*splitratio)
    trainset=[]
    copy=list(dataset)
    while len(trainset)<trainsize:
        index=random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset,copy]
def separatebyclass(dataset):
    separated={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if(vector[-1] not in separated):
            separated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg=mean(numbers)
    variance=sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries=[(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizebyclass(dataset):
    separated=separatebyclass(dataset)
    summaries={}
    for classvalue,instance in separated.items():
        summaries[classvalue]=summarize(instance)
    return summaries

def calculateprobability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent
    
def calculateclassprobability(summaries,inputvector):
    probabilities={}
    for classvalue,classsummaries in summaries.items():
        probabilities[classvalue]=1
        for i in range(len(classsummaries)):
            mean,stdev=classsummaries[i]
            x=inputvector[i]
            probabilities[classvalue]*=calculateprobability(x,mean,stdev)
    return probabilities

def predict(summaries,inputvector):
    probabilities=calculateclassprobability(summaries,inputvector)
    bestlabel,bestprob=None,-1
    for classvalue,probability in probabilities.items():
        if bestlabel is None or probability>bestprob:
            bestprob=probability
            bestlabel=classvalue
    return bestlabel

def getprediction(summaries,testset):
    prediction=[]
    for i in range(len(testset)):
        result=predict(summaries,testset[i])
        prediction.append(result)
    return prediction

def getaccuracy(testset,prediction):
    correct=0
    for i in range(len(testset)):
        #print(testset[i][-1],"",prediction[i])
        if testset[i][-1]==prediction[i]:
            correct+=1
        return(correct/float(len(testset)))*100.0
        
def main():
    filename='irisdata.csv'
    splitratio=0.67
    dataset=loadcsv(filename)
    print("\n the length of the dataset:",len(dataset))
    print("\n the dataset splitting into training and testing\n")
    trainingset,testset=splitdataset(dataset,splitratio)
    print("\n number of rows in training set:{0} rows".format(len(trainingset)))
    print("\n number of rows in training set:{0} rows".format(len(testset)))
    print("\n first five rows of training set:\n")
    for i in range(0,5):
        print(trainingset[i],"\n")
    print("\n first five rows of testing set:\n")
    for i in range(0,5):
        print(testset[i],"\n")
    summaries=summarizebyclass(trainingset)
    print("\n model summarie:\n",summaries)
    prediction=getprediction(summaries,testset)
    print("\n prediction:\n",prediction)
    accuracy=getaccuracy(testset,prediction)
    print("\n accuracy:{0} %".format(accuracy))
main()
    

