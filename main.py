 plt
import tensorflow as tf
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.utils import np_utils


# In[ ]:


TrainingPercent = 80     # Used 80% of dataset for Training
ValidationPercent = 10   # Used 10% of dataset for Validation
TestPercent = 10         # Used 10% of dataset for Test Percent
M = 10                   # Number of Clusters
PHI = []                 # Empty List for Design Matrix
IsSynthetic = False
maxAcc = 0.0     
maxIter = 0
C_Lambda = 0.03          # Initialise Regulariser for Basis Function


# ### Human Observed Data Set

# In[ ]:


diffpair =r"C:\Users\yugal\Downloads\diffn_pairs.csv" 
train_features = pd.read_csv(diffpair)
df_792 = train_features.sample(792,random_state=1)    ##extracting random entries from diffn_pairs (keeping random state fixed)

samepair =r"C:\Users\yugal\Downloads\same_pairs.csv"
train_features = pd.read_csv(samepair)            

output = pd.DataFrame.append(df_792, train_features)   ## merging samepair and diffpair files

feature =r"C:\Users\yugal\Downloads\HumanObserved-Features-Data.csv"
train_features = pd.read_csv(feature)

merge_output = pd.merge(output, train_features, how='left', left_on='img_id_A', right_on='img_id')  ##merging feature file wrt img_id_A 
merge_output.drop(columns = ['Unnamed: 0', 'img_id'], axis =1, inplace = True)  #dropping unwanted columns

merge_output = pd.merge(merge_output, train_features, how='left', left_on='img_id_B', right_on='img_id') ##merging feature file wrt img_id_B 
merge_output.drop(columns = ['Unnamed: 0', 'img_id'], axis =1, inplace = True)  #dropping unwanted columns


#merge_output.columns.values



# ### GSC DATA SET

# In[ ]:


diffpair1 =r"C:\Users\yugal\Downloads\GSC-Features-Data\diffn_pairs.csv" 
train_features1 = pd.read_csv(diffpair1)
df_71532 = train_features1.sample(71532,random_state=1)    ##extracting random entries from diffn_pairs (keeping random state fixed)

samepair1 =r"C:\Users\yugal\Downloads\GSC-Features-Data\same_pairs.csv"
train_features1 = pd.read_csv(samepair1)            

output1 = pd.DataFrame.append(df_71532, train_features1)   ## merging samepair and diffpair files

feature1 =r"C:\Users\yugal\Downloads\GSC-Features-Data\GSC-Features.csv"
train_features1 = pd.read_csv(feature1)

merge_output1 = pd.merge(output1, train_features1, how='left', left_on='img_id_A', right_on='img_id')  ##merging feature file wrt img_id_A 
merge_output1.drop(columns = ['img_id'], axis =1, inplace = True)  #dropping unwanted columns

merge_output1 = pd.merge(merge_output1, train_features1, how='left', left_on='img_id_B', right_on='img_id') ##merging feature file wrt img_id_B 
merge_output1.drop(columns = ['img_id'], axis =1, inplace = True)  #dropping unwanted columns
#merge_output1
#-------------------------------------------------------
merge_concat_gsc=merge_output1
merge_concat_gsc=merge_concat_gsc.loc[:,'target':'f512_y']  
merge_concat_gsc=merge_concat_gsc.sample(frac=1,random_state=1)
merge_concat_gsc_target=merge_concat_gsc.loc[:,'target':'target']
merge_concat_gsc_data=merge_concat_gsc.loc[:,'f1_x':'f512_y'] 
#----------------------------------------------------------
merge_sub_gsc_data=merge_output1

xyz = []
columnToDrop = []
listtt=[]
DropColumnList=[]
for i in range(0,512):
    xyz.append(abs(merge_sub_gsc_data['f'+ str(i+1) +'_x'] - merge_sub_gsc_data['f'+ str(i+1) +'_y']))
    columnToDrop.append('f'+ str(i+1) +'_y') 

merge_sub_gsc_data.drop(columns = columnToDrop, axis =1, inplace = True)  #dropping unwanted columns
merge_sub_gsc_data.drop(columns = ['img_id_A','img_id_B'], axis =1, inplace = True) 

merge_sub_gsc_data=np.asarray(merge_sub_gsc_data)

listtt=np.var(merge_sub_gsc_data,axis=0)
for i in range(0,len(listtt)):
    if listtt[i]==0:
        DropColumnList.append(i)

merge_sub_gsc_data=np.delete(merge_sub_gsc_data,DropColumnList,axis=1)    

#--------------------------------------------------------


#Creating sets 
# 1 merge_concat_gsc_data
# 2 merge_sub_gsc_data
# 3 merge_concat_gsc_target


# ### Creating Data After Subtraction for Human Observed Data-set

# In[ ]:


merge_sub=merge_output
merge_sub['f1_x']=abs(merge_output['f1_x']-merge_output['f1_y'])
merge_sub['f2_x']=abs(merge_output['f2_x']-merge_output['f2_y'])
merge_sub['f3_x']=abs(merge_output['f3_x']-merge_output['f3_y'])
merge_sub['f4_x']=abs(merge_output['f4_x']-merge_output['f4_y'])
merge_sub['f5_x']=abs(merge_output['f5_x']-merge_output['f5_y'])
merge_sub['f6_x']=abs(merge_output['f6_x']-merge_output['f6_y'])
merge_sub['f7_x']=abs(merge_output['f7_x']-merge_output['f7_y'])
merge_sub['f8_x']=abs(merge_output['f8_x']-merge_output['f8_y'])
merge_sub['f9_x']=abs(merge_output['f9_x']-merge_output['f9_y'])


merge_sub=merge_sub.loc[:,'target':'f9_x']    
merge_sub=merge_sub.sample(frac=1,random_state=1)
merge_sub_target=merge_sub
merge_sub_target=merge_sub.loc[:,'target':'target']
merge_sub_data=merge_sub.loc[:,'f1_x':'f9_x']    
#merge_sub_data


# ### Creating Concatenation Data-Set for Human Observed Data

# In[ ]:


merge_concat=merge_output
merge_concat=merge_concat.loc[:,'target':'f9_y']    
merge_concat=merge_concat.sample(frac=1,random_state=1)
merge_concat_target=merge_concat
merge_concat_target=merge_concat.loc[:,'target':'target']
merge_concat_data=merge_concat.loc[:,'f1_x':'f9_y']   


# ### Removing Columns with 0 Variance

# In[ ]:


merge_concat_gsc_data = np.asarray(merge_concat_gsc_data)
#print(merge_concat_gsc_data.shape)
lisss = []
dropColumnList = []

lisss = np.var(merge_concat_gsc_data,axis=0)
for i in range(0,len(lisss)):
    if lisss[i] == 0:
        dropColumnList.append(i)
#print(dropColumnList)
merge_concat_gsc_data = np.delete(merge_concat_gsc_data,dropColumnList,axis=1)
#print(merge_concat_gsc_data.shape)


# ### List to Input Different Datasets

# In[ ]:


data_input_list=[np.asarray(merge_concat_data),np.asarray(merge_sub_data),np.asarray(merge_concat_gsc_data),np.asarray(merge_sub_gsc_data)]
target_input_list=[merge_concat_target,merge_concat_gsc_target]


# ### Functions Used

# In[ ]:


def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):  #Generating Covariance Matrix
    BigSigma    = np.zeros((len(Data),len(Data)))      # Featue * Feature empty matrix
    DataT       = np.transpose(Data)                   # Data x Feature Full data matrix
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))   # Training length     
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])            
        varVect.append(np.var(vct))   # Variance is calculated for each feature
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]+0.002   # Diagonal Matrix with sigma^2(Variance)
    
    BigSigma = np.dot(200,BigSigma)   # Big Sigma value is normalised
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  # To Calculate exponential term in Gaussian Distribution
    R = np.subtract(DataRow,MuRow)  
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))   # Calculating phi matrix value
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80): # Design Matrix
    DataT = np.transpose(Data)
    TrainingLen = math.floor(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    
    BigSigInv = np.linalg.inv(BigSigma)   # Inverse of Big Sigma
    for  C in range(0,len(MuMatrix)):     
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv) 
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda): 
    Lambda_I = np.identity(len(PHI[0])) 
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)   # Transpose of Design Matrix 
    PHI_SQR     = np.dot(PHI_T,PHI)   # PHI(T)xPHI  10x10
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR) # Adding Regularizer
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI) # Inverse of PHI(T)xPHI
    INTER       = np.dot(PHI_SQR_INV, PHI_T) # Inverse(PHI(T)xPHI)xPHI(T) 
    
    W           = np.dot(INTER, T)   # (Inverse(PHI(T)xPHI)xPHI(T))x(Target Value) 
    ##print ("Training Weights Generated..")
    return W

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))  # Returns target data value used for validation and testing
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):    
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)  # (Actual value-Predicted value)^2
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):   # Check number of correctly predicted values
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))     # Return Accuracy = correctly predicted/ total predicted
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))  # return accuracy and erms


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)


def cost_function(features, labels, weights):
    '''
    Cost = ( log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)
    #Take the error when label=1
    labels=np.transpose(labels)
    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    
    class2_cost = (1-labels)*np.log(1-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum()/observations

    return cost


def update_weights(features, labels, weights, lr):
    
    N = len(features)

   
    predictions = predict(features, weights) #Passing through Sigmoid
    
    y=features.T
    t=predictions-labels
    gradient = np.dot(y,t)
    # Take the average cost derivative for each feature
    gradient /= N
    # Multiply the gradient by our learning rate
    gradient *= lr
    # Subtract from our weights to minimize cost
    
    weights -= gradient
    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def classify(preds):
    decision_boundary1 = np.vectorize(decision_boundary)
    return decision_boundary1(preds).flatten()


def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        

    return weights, cost_history


def accuracy(predicted_labels, actual_labels):               #Calculating the Accuracy
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='True')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='False')

    plt.legend(loc='upper right');
    ax.set_title("Graphical Representation of our prediction")
    ax.set_xlabel('Testing Features')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()


# In[ ]:


def LinearRegression(data_input_list,target_input_list):
    for items in range(0,len(data_input_list)):
        merge_sub_data=  data_input_list[items]
        merge_sub_target=target_input_list[math.floor(items/2)]

        # create training and testing vars
        data_train, data_test, target_train, target_test = train_test_split(merge_sub_data,merge_sub_target, test_size=0.2)
        data_test,data_val,target_test,target_val=train_test_split(data_test,target_test, test_size=0.5)

        #-----------------------------------

        ErmsArr = []   # Empty List to store ERMS value
        AccuracyArr = [] #list to store Accuracy
        
        merge_sub_data = np.asarray(np.transpose(merge_sub_data))
        
        data_train = np.asarray(np.transpose(data_train))
        data_test = np.asarray(np.transpose(data_test))
        data_val = np.asarray(np.transpose(data_val))
        
        target_train=np.asarray(target_train)
        target_val=np.asarray(target_val)
        target_test=np.asarray(target_test)

        #-----------------------------------

        kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(data_train)) # K-means Clustering Fitting
        Mu = kmeans.cluster_centers_ #Centroid Values
        print("Mu :")
        print(Mu.shape)
        BigSigma     = GenerateBigSigma(merge_sub_data, Mu, TrainingPercent,IsSynthetic) # Calculate Big Sigma
        print("Big Sigma :")
        print(BigSigma.shape)
        TRAINING_PHI = GetPhiMatrix(merge_sub_data, Mu, BigSigma, TrainingPercent) # Design Matrix for training
        
        print("TRAINING_PHI :")
        print(TRAINING_PHI.shape)
        
        W            = GetWeightsClosedForm(TRAINING_PHI,target_train,(C_Lambda))#Regularisation
        W=np.ravel(W)
        print("W :")
        print(W.shape)
        TEST_PHI     = GetPhiMatrix(data_test, Mu, BigSigma, 100) # Design Matrix for testing
        VAL_PHI      = GetPhiMatrix(data_val, Mu, BigSigma, 100) # Design Matrix for validation
        print("TEST_PHI :")
        print(TEST_PHI.shape)
        print("VAL PHI :")
        print(VAL_PHI.shape)





        TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
        VAL_TEST_OUT = GetValTest(VAL_PHI,W)
        TEST_OUT     = GetValTest(TEST_PHI,W)


        TrainingAccuracy   = str(GetErms(TR_TEST_OUT,target_train)) #erms for training
        ValidationAccuracy = str(GetErms(VAL_TEST_OUT,target_val)) #erms for validation
        TestAccuracy       = str(GetErms(TEST_OUT,target_test)) #erms for testing


        # In[11]:


        print ('UBITname      = Krishna Sehgal')
        print ('Person Number = 50291124')
        print ('----------------------------------------------------')
        print ('----------------------------------------------------')
        print ("-------Closed Form with Radial Basis Function-------")
        print ('----------------------------------------------------')
        print ("M = 10 \nLambda = 0.9")
        print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))    #erms for training
        print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))  #erms for validation
        print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))        #erms for testing
        print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))  #Accuracy for training
        print ("Accuracy Validation = " + str(float(ValidationAccuracy.split(',')[0])))#Accuracy for Validation
        print ("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0]))) #Accuracy for testing


        # ## Gradient Descent solution for Linear Regression

        # In[12]:

        print ('----------------------------------------------------')
        print ('--------------Please Wait for 2 mins!----------------')
        print ('----------------------------------------------------')


        # In[13]:


        W_Now        = np.dot(220, W)  #Initialise weight randomly
        La           = 2               
        learningRate = 0.05
        L_Erms_Val   = []
        L_Erms_TR    = []
        L_Erms_Test  = []
        W_Mat        = []

        Ermstr = []
        Ermsval = []
        Ermstest = []

        for i in range(0,400):

            #print ('---------Iteration: ' + str(i) + '--------------')
            #print("target_train")
            #print(target_train[i].shape)
            #print("W_Now")
            #print(W_Now.shape)
            #print("Training_PHI")
            #print(TRAINING_PHI[i].shape)
            x=np.dot(np.transpose(W_Now),TRAINING_PHI[i])
            y=(target_train[i] - x)[0]
            z=TRAINING_PHI[i]
            Delta_E_D     = -np.dot(y,z)              
            La_Delta_E_W  = np.dot(La,W_Now)
            Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
            Delta_W       = -np.dot(learningRate,Delta_E)
            W_T_Next      = W_Now + Delta_W
            W_Now         = W_T_Next

            #-----------------TrainingData Accuracy---------------------#
            TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
            Erms_TR       = GetErms(TR_TEST_OUT,target_train)
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))

            #-----------------ValidationData Accuracy---------------------#
            VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
            Erms_Val      = GetErms(VAL_TEST_OUT,target_val)
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))

            #-----------------TestingData Accuracy---------------------#
            TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
            Erms_Test = GetErms(TEST_OUT,target_test)
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))

        #-----------------------------------    

        print ('----------Gradient Descent Solution--------------------')
        print ("M = 15 \nLambda  = 0.0001\neta=0.01")
        print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
        print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
        print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
        #-----------------------------------  
        plt.xlim(1,400)
        plt.plot(L_Erms_TR)
        plt.xlabel("Iterations")
        plt.ylabel("Training ERMS Values")
        plt.show()

        plt.xlim(1,400)
        plt.plot(L_Erms_Val)
        plt.xlabel("Iterations")
        plt.ylabel("Validation ERMS Values")
        plt.show()

        plt.xlim(1,400)
        plt.plot(L_Erms_Test)
        plt.xlabel("Iterations")
        plt.ylabel("Testing ERMS Values")
        plt.show()





# In[ ]:


def LogisticRegression(data_input_list,target_input_list):
    for items in range(0,len(data_input_list)):
        merge_sub_data=  data_input_list[items]
        merge_sub_target=target_input_list[math.floor(items/2)]

        # create training and testing vars
        data_train, data_test, target_train, target_test = train_test_split(merge_sub_data,merge_sub_target, test_size=0.2)
        
        #-----------------------------------
        
        iters=10000
        lr=0.05
        true=[]
        false=[]
        target_test=np.asarray(target_test)
    
        merge_sub_data = np.asarray(np.transpose(merge_sub_data)) 
        data_train = np.asarray(data_train)
        data_test = np.asarray(data_test)
        print(data_test.shape)
        
        weight = np.ones(data_train.shape[1])
        weight=np.mat(weight)
        weight=np.transpose(weight)
        
        target_train=np.asarray(target_train)
        target_test=np.asarray(target_test)
        
        

        #-----------------------------------
        weights, cost_history=train(data_train, target_train, weight, lr, iters)   #Function Call for Training Model
        
        prediction_test=np.dot(data_test,weights)  #Prediction on Testing Data
        
        prediction_test=sigmoid(prediction_test)   # Sigmoid Function Call
        for item in prediction_test:               # Storing Data in two list to plot graph  
            if item>=0.5:
                true.append(item)
            else:
                false.append(item)

        
        plot_decision_boundary(true, false)
        Normalise=np.asarray(classify(prediction_test))
        Normalise=np.transpose(Normalise)
        Accuracy=accuracy(Normalise,target_test)*100
        
        print ('UBITname      = Krishna Sehgal')
        print ('Person Number = 50291124')
        print ('----------------------------------------------------')
        print ('----------------------------------------------------')
        print ("-------Logistic Regression-------")
        print ('----------------------------------------------------')
        print ('----------Gradient Descent Solution--------------------')
        print ("Learning Rate  = 0.05")
        print ("Accuracy: ")
        print (Accuracy) #Accuracy for testing
       


# In[ ]:


LogisticRegression(data_input_list,target_input_list)


# In[ ]:


LinearRegression(data_input_list,target_input_list)


# In[ ]:


def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == 0):
            # Does not match
            processedLabel.append([1])   
        else:
            # Match
            processedLabel.append([0])  
            
    return  np_utils.to_categorical(np.array(processedLabel),2)

def processData(dataset):
    labels = dataset
    processedLabel = encodeLabel(labels) 
    return processedLabel

def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return 0
    elif encodedLabel == 1:
        return 1


def get_model():
    
    
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid'))
    
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    
    
    model.summary()
    
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np

#data_input_list=np.asarray(data_input_list)
print(len(data_input_list[0][0]))

drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 2
validation_data_split = 0.2

model_batch_size = 128
tb_batch_size = 32
early_patience = 100
num_epochs = 200
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
input_size = 0


for items in range(0,len(data_input_list)):
    
    wrong   = 0
    right   = 0
    input_size = data_input_list[items].shape[1]
    model = get_model()
    merge_sub_data=  data_input_list[items]
    merge_sub_target=target_input_list[math.floor(items/2)]
    # create training and testing vars
    data_train, data_test, target_train, target_test = train_test_split(merge_sub_data,merge_sub_target, test_size=0.2)
    #---------------------------------
    
    data_train=np.asarray(data_train)
    data_test=np.asarray(data_test)
    target_train=np.asarray(target_train)
    target_test=np.asarray(target_test)
    

    #---------------------------
    
    processedTrainingLabel = processData(target_train) # Function call 
    processedTestingLabel  = processData(target_test)

    #---------------------------

    processedTestData  = data_test
    processedTestLabel = processedTestingLabel
    predictedTestLabel = []


    # Process Dataset
    processedData, processedLabel = data_train,processedTrainingLabel
    validation_data_split = 0.2
    num_epochs = 200
    model_batch_size = 128
    history = model.fit(processedData, processedLabel, validation_split=validation_data_split, epochs=num_epochs, batch_size=model_batch_size, callbacks = [tensorboard_cb,earlystopping_cb])


    get_ipython().run_line_magic('matplotlib', 'inline')
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))   


    for i,j in zip(processedTestData,processedTestLabel):
        y = model.predict(np.array(i).reshape(-1,input_size))
        predictedTestLabel.append(decodeLabel(y.argmax()))

        if j.argmax() == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1

    print("Errors: " + str(wrong), " Correct :" + str(right))
    print ('UBITname      = Krishna Sehgal')
    print ('Person Number = 50291124')
    print ('-----------------------------')
    print ("-------Neural Networks-------")
    print ('-----------------------------')
    print("Testing Accuracy: " + str(right/(right+wrong)*100))

