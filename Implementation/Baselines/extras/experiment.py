import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import glob
import torch
import sys
import pandas as pd
import transformers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text
from bert import bert_tokenization
from scipy.spatial import distance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM

class model:
  def __init__(self,df,model_name):
    self.tokenizer=None
    self.model=None
    self.tokenized_padded_text=None
    self.attention_mask=None
    self.textip=None
    self.pooledOp=None
    self.input_dfs=None
    self.data_frame=df
    self.feature_df=None
    self.model_name=None
    self.InitModel(model_name)

  def InitModel(self,model_name) :
    if model_name == 'distilBert':
      model_class, tokenizer_class, pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')  
      self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
      self.model = model_class.from_pretrained(pretrained_weights)
      self.model_name='distilBert'
    if model_name == 'mBert':
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
      self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
      self.model_name='mBert'
    if model_name=='muril':
      self.textip = tf.keras.layers.Input(shape=(), dtype=tf.string)
      self.max_seq_length = 128
      muril_model, muril_layer =self.init_muril(model_url="https://tfhub.dev/google/MuRIL/1", max_seq_length=self.max_seq_length)
      vocab_file = muril_layer.resolved_object.vocab_file.asset_path.numpy()
      do_lower_case = muril_layer.resolved_object.do_lower_case.numpy()
      self.tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
      self.model_name='muril'
      self.model=muril_model
  
  def tokenize(self,column):
       tokenized_text=column.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
       # Padding
       max_len = 0
       for i in tokenized_text.values:
        if len(i) > max_len:
         max_len = len(i)

       self.tokenized_padded_text = np.array([i + [0]*(max_len-len(i)) for i in tokenized_text.values])
       self.create_attention_mask()

  def create_attention_mask(self):
      self.attention_mask = np.where(self.tokenized_padded_text != 0, 1, 0)
      self.input_ids = torch.tensor(self.tokenized_padded_text)  
      self.attention_mask = torch.tensor(self.attention_mask)

  def init_muril(self,model_url, max_seq_length):
    inputs = dict(
    input_word_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
    input_mask=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
    input_type_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
    )

    muril_layer = hub.KerasLayer(model_url, trainable=True)
    outputs = muril_layer(inputs)

    assert 'sequence_output' in outputs
    assert 'pooled_output' in outputs
    assert 'encoder_outputs' in outputs
    assert 'default' in outputs
    return tf.keras.Model(inputs=inputs,outputs=outputs["encoder_outputs"]), muril_layer
  
  def create_input(self,input_strings, tokenizer, max_seq_length):
     input_ids_all, input_mask_all, input_type_ids_all = [], [], []
     for input_string in input_strings:
       input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
       input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
       sequence_length = min(len(input_ids), max_seq_length)
    
       if len(input_ids) >= max_seq_length:
        input_ids = input_ids[:max_seq_length]
       else:
        input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

       input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

       input_ids_all.append(input_ids)
       input_mask_all.append(input_mask)
       input_type_ids_all.append([0] * max_seq_length)
  
     return np.array(input_ids_all), np.array(input_mask_all), np.array(input_type_ids_all)

  def encode(self,input_text):
      input_ids, input_mask, input_type_ids = self.create_input(input_text, 
                                                       self.tokenizer, 
                                                       self.max_seq_length)
      inputs = dict(
       input_word_ids=input_ids,
       input_mask=input_mask,
       input_type_ids=input_type_ids,
      )
      return self.model(inputs)

  def GetFeatures(self,input=None):
   if self.model_name!='muril':
      with torch.no_grad():
       last_hidden_states = self.model(self.input_ids, attention_mask=self.attention_mask)
      last_hidden_states['last_hidden_state'].size()   
      self.features = last_hidden_states[0][:,0,:].numpy()
      self.features=pd.DataFrame(self.features)
   elif self.model_name=='muril':
      embeddings = self.encode(input)
      f=embeddings[11][:,0,:]
      self.features=pd.DataFrame(f.numpy())
   return self.features

class classifiers:

  def __init__(self,features,label):
    self.features_set=features
    self.labels=label
    self.Createstaticsplit(features,label)
    self.accuracy=list()
    self.f1score=list()
    self.models=list()
    self.y_pred=list()

  def classify(self,svm=True,random_forest=True,xgboost=True,logistic_regression=True,ann=True)  :
      if svm==True:
        acc,f1_score=self.CreateSVMClassifier()
        self.accuracy.append(acc)
        self.f1score.append(f1_score)
        
        self.models.append('svm')
      if random_forest==True:
        acc,f1_score,y=self.RandomForestClassifier()
        self.accuracy.append(acc)
        self.f1score.append(f1_score)
        self.y_pred.append(y)
        self.models.append('random_forest')
      if xgboost==True:
        acc,f1_score,y_pred=self.XGBClassifier(2)
        self.accuracy.append(acc)
        self.f1score.append(f1_score)
        self.y_pred.append(y)
        self.models.append('xgboost')
      if logistic_regression==True:
        acc,f1_score,y=self.LogisticRegression()
        self.accuracy.append(acc)
        self.f1score.append(f1_score)
        self.y_pred.append(y)
        self.models.append('lr')
      if ann==True:
        acc,f1_score=self.annClassifier()
        self.accuracy.append(acc)
        self.f1score.append(f1_score)
        self.models.append('ann')
      return self.accuracy,self.f1score,self.models,self.y_pred

  def MajorityVotingClassifier(self, num_class):
        acc_xg, f1_xg, y_xgboost = self.XGBClassifier(num_class)
        acc_rf, f1_rf, y_rf = self.RandomForestClassifier()
        acc_rf, f1_rf, y_lr = self.LogisticRegression()
        y_pred = list()
        for i in range(len(y_xgboost)):
            preds = list()
            preds.append(y_xgboost[i])
            preds.append(y_rf[i])
            preds.append(y_lr[i])
            y_pred.append(max(set(preds), key=preds.count))
        cm=confusion_matrix(self.test_labels,y_pred)    
        return accuracy_score(self.test_labels, y_pred), f1_score(self.test_labels, y_pred, average='macro') ,cm,y_pred 

  def XGBClassifier(self,num_class):
     from xgboost import XGBClassifier
     classifier = XGBClassifier(n_estimators=500,learning_rate=1, max_depth=2,objective='multi:softmax',num_class=num_class)
     classifier.fit(self.train_features, self.train_labels)
     
     y_pred = classifier.predict(self.test_features)
     return accuracy_score(self.test_labels, y_pred),f1_score(self.test_labels, y_pred,average='macro'),y_pred

  def CreateSVMClassifier(self):
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'poly',decision_function_shape='ovr', random_state = 0)
    classifier.fit(self.train_features, self.train_labels)
  
    y_pred = classifier.predict(self.test_features)
    return accuracy_score(self.test_labels, y_pred),f1_score(self.test_labels, y_pred,average='macro')

  def RandomForestClassifier(self):
     from sklearn.ensemble import RandomForestClassifier
     classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
     classifier.fit(self.train_features, self.train_labels)

     y_pred = classifier.predict(self.test_features)
     return accuracy_score(self.test_labels, y_pred),f1_score(self.test_labels, y_pred,average='macro'),y_pred

  def Createstaticsplit(self,features,labels,split_per=0.8):
   num=np.shape(features)[0]
   self.train_features=features.head(int(split_per*num))
   self.train_labels=labels.head(int(split_per*num))
   self.test_features=features.tail(num-int(split_per*num))
   self.test_labels=labels.tail(num-int(split_per*num))
  def annClassifier(self):
      import tensorflow as tf
      from sklearn.compose import ColumnTransformer
      from sklearn.preprocessing import OneHotEncoder
      ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
      feature_set = np.array(ct.fit_transform(self.features_set))
      train_features, test_features, train_labels, test_labels = train_test_split(feature_set, self.labels)

      ann = tf.keras.models.Sequential()
      ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
      ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
      ann.add(tf.keras.layers.Dense(units=1, activation='softmax'))
      ann.add(tf.keras.layers.Dense(units=1, activation='softmax'))
      ann.add(tf.keras.layers.Dense(units=1, activation='softmax'))
      ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
      ann.fit(train_features, train_labels, batch_size = 32, epochs = 200)

      y_pred = ann.predict(test_features)
      return accuracy_score(test_labels, y_pred),f1_score(test_labels, y_pred,average='macro')

  def LogisticRegression(self):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression 
    lr_clf = LogisticRegression(multi_class='multinomial')
    lr_clf.fit(self.train_features, self.train_labels) 

    y_pred=lr_clf.predict(self.test_features)
    return accuracy_score(self.test_labels, y_pred),f1_score(self.test_labels, y_pred,average='macro'),y_pred

def ReadData(path):
 path=path+"DatasetLanguageWise\\*.pkl" 
 all_files = sorted(glob.glob(path))
 print(all_files)
 li = []
 labels=[]
 for filename in all_files:
    df = pd.read_pickle(filename)
    li.append(df)
    labels.append(df.label)

 dataset = pd.concat(li, axis=0, ignore_index=True)
 labels=pd.concat(labels,axis=0,ignore_index=True)
 return dataset,labels

def ExtractFeatures(dataset,labels,model):
  model_pipeline=model(dataset,model)
  model_pipeline.tokenize(dataset.commentText)
  feature_df_mbert=model_pipeline.GetFeatures(dataset.commentText)
  feature_df_mbert['label']=labels
  return feature_df_mbert

def SaveAsCsv(dataframe,modelname):
    filename='CD_'+str(modelname)+'.csv'
    dataframe.to_csv(filename,index=False)

def classify(dataframe,classifier):
   classifier_class=classifiers(features=dataframe.iloc[:, :-1],label=dataframe.label)
   test_labels=dataframe.label.tail(len(dataframe)-int(0.8*len(dataframe)))
   if classifier == 'A':
    accuracies,f1_scores,models,y=classifier_class.classify(svm=True,random_forest=True,xgboost=True,logistic_regression=True,ann=False)
    for i in range(len(y)) :
      print(models[i])
      print(classification_report(test_labels, y[i], labels=[0,1]))
    a,f,cm,y_pred=classifier_class.MajorityVotingClassifier(2)  
    print("VC")
    print(classification_report(test_labels, y_pred, labels=[0,1]))   
   elif classifier == 'LR':
      accuracies,f1_scores,models,y=classifier_class.classify(svm=False,random_forest=False,xgboost=False,logistic_regression=True,ann=False)
      print(classification_report(test_labels, y[0], labels=[0,1]))
   elif classifier == 'RF':  
      accuracies,f1_scores,models,y=classifier_class.classify(svm=False,random_forest=True,xgboost=False,logistic_regression=False,ann=False)
      print(classification_report(test_labels, y[0], labels=[0,1]))
   elif classifier == 'XGBOOST':
      accuracies,f1_scores,models,y=classifier_class.classify(svm=False,random_forest=False,xgboost=True,logistic_regression=False,ann=False)
      print(classification_report(test_labels, y[0], labels=[0,1]))
   elif classifier == 'VC': 
      a,f,cm,y_pred=classifier_class.MajorityVotingClassifier(2)  
      print(classification_report(test_labels, y_pred, labels=[0,1]))

def main():
    args = sys.argv[1:]
    dataset,labels=ReadData('')
    model=args[0]
    classifier=args[1]
    dataframe=ExtractFeatures(dataset,labels,model)
    SaveAsCsv(dataframe,model)
    classify(dataframe,classifier)

if __name__ == "__main__":
    main()
