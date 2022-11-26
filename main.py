import random
import os


class Perceptron:
    weights = []
    lr = 0
    theta = 1

    def __init__(self, weightsNum:int, lr:float):
        self.weights = [random.uniform(0,1) for i in range(weightsNum)]
        self.lr = lr

    def predict(self, example:list)->float:
        dotproduct = 0
        for i in range(len(self.weights)):
            dotproduct += example[i]*self.weights[i]
        return dotproduct > self.theta

    def adjustWeight(self, example:list, true:int):
        pred = self.predict(example)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.lr*(true-pred)*example[i]
        self.theta = self.theta - self.lr*(true-pred)
        
    def trainForEpoch(self, trainset:dict):
        keys = list(trainset.keys())
        random.shuffle(keys)
        for i in keys:
            self.adjustWeight(list(i),trainset.get(i))
            
    def getAccuracy(self, testset:dict)->float:
        sum = 0
        for i in testset.keys():
            if(int(self.predict(list(i))) == testset.get(i)):
                sum += 1
        return (sum/len(testset.keys()))*100
        

class Network:

    perceptrons = []

    def __init__(self, numofclasses, lr):
        for i in range(numofclasses):
            self.perceptrons.append(Perceptron(26,lr))
    
    def predict(self, example):
        predictions = []
        for p in self.perceptrons:
            predictions.append(p.predict(example))
        return predictions
    
    def adjustWeights(self, example, true):
        predictions = self.predict(example)
        mapTrue = []
        for j in range(len(self.perceptrons)):
            if j == true:
                mapTrue.append(1)
            else:
                mapTrue.append(0)
        for i in range(len(self.perceptrons)):
            self.perceptrons[i].adjustWeight(example,mapTrue[i])

    def trainForEpoch(self, trainset):
        keys = list(trainset.keys())
        random.shuffle(keys)
        for i in keys:
            self.adjustWeights(list(i),trainset.get(i))

    def getAccuracy(self, testset:dict)->float:
        sum = 0
        for i in testset.keys():
            if(self.predict(i).count(1)>0):
                if(self.predict(i).index(1) == testset.get(i)):
                    sum += 1
        return (sum/len(testset.keys()))*100



def readData(dataDir:str):
    data = {}
    for dir in os.listdir(dataDir):
        for filename in os.listdir(os.path.join(dataDir,dir)):
            if(os.path.isfile(os.path.join(dataDir,dir,filename))):
                file = open(os.path.join(dataDir,dir,filename), encoding='utf8')
                vec = [0 for x in range(26)]
                for line in file.readlines():
                    l = line.strip().replace(" ","").lower()
                    for c in l:
                        if ord(c) >= 97 and ord(c) <=122:
                            vec[ord(c)-97] +=1
                file.close()
                sumOfVec = sum(vec)
                for i in vec:
                    i = i/sumOfVec
                data[tuple(vec)] = dir
    return data

def main():
    trainset = readData('lang')
    testset = readData('lang.test')
    labels = list(set(trainset.values()))
    labelMaping = {}
    for i in range(len(labels)):
        labelMaping[labels[i]] = i
        
    for k1 in trainset.keys():
        trainset[k1] = labelMaping[trainset[k1]]
        
    for k2 in testset.keys():
        testset[k2] = labelMaping[testset[k2]]
    
    lr = 0.01
    model = Network(len(labels),lr)
    
    epochs = 5
    for i in range(epochs):
        model.trainForEpoch(trainset)
        print(f"\nafter epoch {i+1}")
        print(f'accuracy on test: {model.getAccuracy(testset)}')
    
    x = input("input new example: ")
    while(len(x.split())>1):
        y = x.strip().replace(" ","").lower()
        yList = [0 for x in range(26)]
        for c in y:
            if ord(c) >= 97 and ord(c) <=122:
                yList[ord(c)-97] +=1
        sumYList = sum(yList)
        for j in yList:
            j = j/sumYList
        prediction = model.predict(yList)
        print(list(labelMaping.keys())[prediction.index(1)])
        x = input("input new example: ")

if __name__ == '__main__':
    main()