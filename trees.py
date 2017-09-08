# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:55:36 2017
@author: Administrator
"""


from math import log
import operator

'''
###################
#### 决策树    ####
###################
'''

# 3-1
# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):  
    numEntries=len(dataSet)   #数据集中实例的总数
    labelCounts={}            #标签字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  #实例的最后一项表示标签，所以取[-1]
        if currentLabel not in labelCounts.keys(): #如果字典中没有该类标签
            labelCounts[currentLabel]=0            #创建以该标签为键的字典项，初始键值为0
        labelCounts[currentLabel] += 1  #字典中对应标签的键值加1
    shannonEnt = 0.0
    for key in labelCounts:    #取字典中的每一个“键—值”对，
        prob = float(labelCounts[key])/numEntries  #prob=该标签的键值/实例总数
        shannonEnt -=prob*log(prob,2)         #以2为底求对数
    return shannonEnt

'''
#################################################
###        测试calcShannonEnt(dataSet)函数   #####
#################################################
def creatDataSet():                             #
    dataSet=[[1,1,'y'],                         #
             [1,1,'y'],                         #
             [1,0,'n'],                         #
             [0,1,'n'],                         #
             [0,1,'n']]                         #
    labels=['no surfacing','flippers']          #
    return dataSet,labels                       #
                                                # 
mydata,mylable=creatDataSet()                   #
print calcShannonEnt(mydata)                    #
                                                #
#################################################                    
'''

# 3-2
# 按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):  #dataSet是原始数据集，每一个元素都是一个包含各属性值的完整信息
    retDataSet = []          #retDataSet分类后返回的列表，其中的元素不再包括已被使用过的属性
    for featVec in dataSet:  #对于原始数据集中的第featVec个元素
        if featVec[axis] == value:           #如果元素的第axis个属性值为value，则该元素被选取
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])   
            retDataSet.append(reduceFeatVec)       #以上三步：将符合条件的元素插入返回列表中，但元素中不再包含已被使用过的元素
    return retDataSet

'''
#################################################
### 测试splitDataSet(dataSet,axis,value)函数   ##
#################################################
def creatDataSet():                             #
    dataSet=[[1,1,'y'],                         #
             [1,1,'y'],                         #
             [1,0,'n'],                         #
             [0,1,'n'],                         #
             [0,1,'n']]                         #
    labels=['no surfacing','flippers']          #
    return dataSet,labels                       #
                                                # 
mydata,mylable=creatDataSet()                   #                   
print splitDataSet(mydata,0,0)                  #
                                                #                              
################################################# 
'''

# 3-3
# 选择最好的数据集划分方式
# 它的划分方式是将每个属性的每个值都作为一次划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       #获取属性个数
    bestEntrory = calcShannonEnt(dataSet)   #调用calcShannonEnt()函数计算原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):   #遍历所有属性
        featList = [example[i] for example in dataSet]  #获取所有元素的第i个属性，返回featList列表
        uniqueVals = set(featList) #将featList由列表转为无重复元素的集合形式
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)     #得到由（属性i的值=value）作为划分方式的划分数据集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) #叠加得到计算由（属性i的值=value）作为划分方式得到的数据集的香农熵
        infoGain = bestEntrory - newEntropy   #计算信息增益
        if (infoGain > bestInfoGain):   #比较划分前后信息增益，  
           bestInfoGain = infoGain      #更新目前最好的信息增益
           bestFeature = i              #更新目前最好的划分属性
    return bestFeature


'''
#################################################
### 测试chooseBestFeatureToSplit(dataSet)函数   ##
#################################################
def creatDataSet():                             #
    dataSet=[[1,1,'y'],                         #
             [1,1,'y'],                         #
             [1,0,'n'],                         #
             [0,1,'n'],                         #
             [0,1,'n']]                         #
    labels=['no surfacing','flippers']          #
    return dataSet,labels                       #
                                                #
mydata,mylable=creatDataSet()                   #
print chooseBestFeatureToSplit(mydata)          #
################################################# 
'''

# 3-3-1
# 寻找出现次数最多的类名

def majorityCnt(classList):   #类名列表classList
    classCount={}             #创建空字典
    for vote in classList:    #以类名作为键，并统计键值（类名出现的次数）
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) 
    #对结果字典进行降序排序，结果返回列表
    return sortedClassCount[0][0]  #选取出现次数最多的类名，即第0行，第0列
                      
# 3-4
# 创建树的函数代码
def createTree(dataSet,labels):  #labels是属性列表
    classList = [example[-1] for example in dataSet]    #创建类列表
    if classList.count(classList[0]) == len(classList): #类列表中只有单一的类
        return classList[0]                             #则停止划分
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)   #遍历所有类名，返回出现次数最多的类名
        #dataSet是在不停的递归中减少属性的
        #具体原因见倒数第二行代码的createTree(splitDataSet())
        #splitDataSet返回的变量在递归过程中作为createTree的第一个变量dataSet
        #所以dataSet长度为1，表示只剩下类标签了，也就是已经使用完了所有的属性    
    bestFeat = chooseBestFeatureToSplit(dataSet)  #获取最好的划分属性在属性列表中的序号
    bestFeatLabel = labels[bestFeat]              #获取最好的划分属性的名字
    myTree = {bestFeatLabel:{}}   #创建以属性为键的嵌套字典
    del(labels[bestFeat])         #从属性列表中删除已用过的属性
    featValues = [example[bestFeat] for example in dataSet]  #获取最好属性的各种取值
    uniqueVals = set(featValues)  #建立最好属性的取值的无重复列表
    for value in uniqueVals:      #遍历最好属性的各种取值
        subLabels = labels[:]     #获取属性列表中现存的属性
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
        #以当前最好属性的取值value建立子树，进入递归
        #最后myTree称为一个嵌套的字典
    return myTree

'''
#################################################
###    测试createTree(dataSet,labels)函数      ##
#################################################
def creatDataSet():                             #
    dataSet=[[1,1,'y'],                         #
             [1,1,'y'],                         #
             [1,0,'n'],                         #
             [0,1,'n'],                         #
             [0,1,'n']]                         #
    labels=['no surfacing','flippers']          #
    return dataSet,labels                       #
                                                #
mydata,mylable=creatDataSet()                   #
mytree = createTree(mydata,mylable)             #
print mytree                                    #
#################################################   
'''      


# 3-5
# 使用文本注释绘制树节点
import matplotlib
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth",fc="0.8")
#定义结点的格式
#是以字典形式给出的，boxstyle为文本框的类型，sawtooth是锯齿形；fc是边框线粗细 
#也可以写成decisionNode = {boxstyle:"sawtooth",fc:"0.8"}
#以下类似
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")                   
                                                     

def plotNode(nodeTxt,centerPt,parentPt,nodeType):      #绘制“带箭头的注解”
      #参数的含义依次为（添加的注释的具体内容，注释需要的文本的坐标（箭头结束的位置），需要被注释的点的坐标（箭头开始的位置））
    createPlot.ax1.annotate(nodeTxt,                   #添加的注释的具体内容
                            xy=parentPt,                #需要注释的点的坐标
                            xycoords='axes fraction',  #
                            xytext=centerPt,           #注释需要的文本的坐标
                            textcoords='axes fraction',#
                            va="center",
                            ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)
'''   
def createPlot():
    fig = plt.figure(1,facecolor='white')  #创建一个画布，背景是白色
    fig.clf()                              #清空画布
    createPlot.ax1 = plt.subplot(111,frameon=False)
    # createPlot.ax1为全局变量，绘制图像的句柄
    # subplot为定义了一个绘图，111表示figure中的图有1行1列，第1个
    # frameon表示是否绘制坐标轴矩形 
    plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode) 
    plt.show()
'''  

'''
###################################
###   测试createPlot()函数      ##
################################### 
                                  #
createPlot()                      # 
                                  #
###################################              
'''  

# 3-6
# 获取叶节点的数目和树的层数
def getNumberLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0] 
    # myTree.keys()返回字典所有“键名”形成的列表
    # 所以myTree.keys()[0] 表示这个列表中第一个“键名”
    secondDict = myTree[firstStr]
    # 获取“键名”为firstStr的“子字典”的内容。（因为 myTree是嵌套的字典）
    for key in secondDict.keys():  #判断字典中是否还嵌套字典
        if type(secondDict[key]).__name__=='dict':           
            numLeafs += getNumberLeafs(secondDict[key])  
            # 如果嵌套字典，则继续递归判断
        else:            
            numLeafs += 1   #不再嵌套，则得到一个叶节点            
    return numLeafs

def getTreeDepth(myTree):  
    #判断树的深度与上面方法类似，递归判断是否还有嵌套字典
    #最深的字典即为最深的树的深度
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
                     
'''
#######################################################################
#####      测试 getNumberLeafs()和getTreeDepth)函数             ########
#######################################################################     
def retrieveTree(i):                                                  #
    listofTree =[{'no surfacing': {0: 'n', 1: {'flippers':            #
                                         {0: 'n', 1: 'y'}}}},         #
                 {'no surfacing': {0: 'n', 1: {'flippers':            #
                    {0: {'head':{0:'no',1:'y'}}, 1: 'n'}}}}           #
                ]                                                     #
    return listofTree[i]                                              #
                                                                      #
print retrieveTree(1)                                                 #
myTree = retrieveTree(0)                                              #
print getNumberLeafs(myTree)                                          #
print getTreeDepth(myTree)                                            #
#######################################################################
'''        

# 3-7
# 绘制树
def plotMidText(cntrPt,parentPt,txtString): #计算子节点和父节点的中心位置，在此位置添加属性可取的不同值
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0] 
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1] 
    createPlot.ax1.text(xMid,yMid,txtString)
    #.text()添加文字说明，参数依次是（横坐标、纵坐标、文字）

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumberLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]  #第一个结点（属性）的名字
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    #plotTree.xOff和plotTree.yOff用于追综已绘制的结点的位置
    #
    plotMidText(cntrPt,parentPt,nodeTxt)             #调用plotMidText（）绘制属性的取值，即由这个属性值决定的分支
    plotNode(firstStr,cntrPt,parentPt,decisionNode)  #调用plotNode（）绘制决策结点
    secondDict = myTree[firstStr]  #获取属性firstStr对应的键值
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #进入下一层，y坐标提前更新
    for key in secondDict.keys():   #对下一层的属性进行判断
        if type(secondDict[key]).__name__=='dict':     #是字典
            plotTree(secondDict[key],cntrPt,str(key))  #进入建树的递归中
        else:  #不是字典，那就是确定的键值了，也就是叶子了，那就绘制叶节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff+1.0/plotTree.totalD  #更新y坐标
    
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')  #创建一个画布，背景是白色
    fig.clf()                              #清空画布
    axprops = dict(xticks=[],yticks=[])
    # xticks和yticks为x,y轴的主刻度和次刻度设置颜色、大小、方向，以及标签大小。
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    # createPlot.ax1为全局变量，绘制图像的句柄
    # subplot为定义了一个绘图，111表示figure中的图有1行1列，第1个
    # frameon表示是否绘制坐标轴矩形     
    plotTree.totalW = float(getNumberLeafs(inTree)) #全局变量，树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))   #全局变量，书的高度
    plotTree.xOff = -0.5/plotTree.totalW #初始横坐标
    plotTree.yOff = 1.0                   #初始纵坐标
    plotTree(inTree,(0.5,1.0),'')        #绘制
    plt.show()                           #展示
        
'''
#######################################################################
#####      测试 3-7 绘制树                                      ########
#######################################################################     
def retrieveTree(i):                                                  #
    listofTree =[{'no surfacing': {0: 'n', 1: {'flippers':            #
                                         {0: 'n', 1: 'y'}}}},         #
                 {'no surfacing': {0: 'n', 1: {'flippers':            #
                    {0: {'head':{0:'no',1:'y'}}, 1: 'n'}}}}           #
                ]                                                     #
    return listofTree[i]                                              #
myTree = retrieveTree(0)                                              #
myTree['no surfacing'][3]='maybe'                                     #
createPlot(myTree)                                                    # 
#######################################################################  
'''        
# 3-8
# 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):  
    #input为已经训练好的训练集，并且以字典形式存储
    #featLabels为属性集
    firstStr = inputTree.keys()[0]    #获取用于分类的第一个属性
    secondDict = inputTree[firstStr]  #获取第二层字典，即第一个属性的“键值”
    featIndex = featLabels.index(firstStr)  #查询用于分类的第一个属性在属性集中的位置
    for key in secondDict.keys():     #遍历第二层字典
        if testVec[featIndex] == key:  
            #如果测试样例的第featIndex属性的值=key，则继续进行下列判断
            #否则退出if，进入下一层for循环，直到找到与之对应的属性
            if type(secondDict[key]).__name__=='dict':  #并且在第二层字典中该属性依旧是字典
                classLabel = classify(secondDict[key],featLabels,testVec)  #则进入递归
            else:
                classLabel = secondDict[key]            #如果不是字典，则可以进行分类，分类结束
    return classLabel

'''
#######################################################################
#####                     测试 3-8 classify()函数              ########
####################################################################### 
def creatDataSet():                                                   #
    dataSet=[[1,1,'y'],                                               #
             [1,1,'y'],                                               #                        
             [1,0,'n'],                                               #                         
             [0,1,'n'],                                               #                        
             [0,1,'n']]                                               #
    labels=['no surfacing','flippers']                                #          
    return dataSet,labels                                             #
def retrieveTree(i):                                                  #
    listofTree =[{'no surfacing': {0: 'n', 1: {'flippers':            #
                                         {0: 'n', 1: 'y'}}}},         #
                 {'no surfacing': {0: 'n', 1: {'flippers':            #
                    {0: {'head':{0:'no',1:'y'}}, 1: 'n'}}}}           #
                ]                                                     #
    return listofTree[i]                                              #
myTree = retrieveTree(0)                                              # 
mydata,mylable=creatDataSet()                                         # 
                                                                      # 
print mylable                                                         # 
print myTree                                                          # 
print classify(myTree,mylable,[1,0])                                  # 
print classify(myTree,mylable,[1,1])                                  #
####################################################################### 
'''

# 3-9
# 使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

'''
#######################################################################
#####                     测试 3-9 存储决策树                    ########
####################################################################### 
def retrieveTree(i):                                                  #
    listofTree =[{'no surfacing': {0: 'n', 1: {'flippers':            #
                                         {0: 'n', 1: 'y'}}}},         #
                 {'no surfacing': {0: 'n', 1: {'flippers':            #
                    {0: {'head':{0:'no',1:'y'}}, 1: 'n'}}}}           #
                ]                                                     #
    return listofTree[i]                                              #
myTree = retrieveTree(0)                                              #   
storeTree(myTree,'classifierStorage.txt')                             #
print grabTree('classifierStorage.txt')                               #
#######################################################################
'''    
    
# 3-10
# 预测隐形眼镜类型
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
print lensesTree
createPlot(lensesTree)





        

    