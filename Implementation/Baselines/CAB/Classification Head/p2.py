# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# %%
import numpy as np
import pandas as pd

# %%
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None) 

# %%
df = pd.read_pickle("train_p2_flattened.pkl")
test = pd.read_pickle("test_p2_flattened.pkl")

# %%
df["list_context"] = df["context"].map(lambda a: a.split("[SEP]"))
test["list_context"] = test["context"].map(lambda a: a.split("[SEP]"))

# %%
df = df.drop(['context'], axis = 1)
df = df.drop(['tweet_id'], axis = 1)

# %%
test = test.drop(['context'], axis = 1)
test = test.drop(['tweet_id'], axis = 1)

# %%
import numpy as np
import torch
import pandas as pd
import transformers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text
from bert import bert_tokenization
from scipy.spatial import distance
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM

# %%
class model:
    def __init__(self, df, model_name, avg_pooling=False):
        self.tokenizer = None
        self.model = None
        self.tokenized_padded_text = None
        self.attention_mask = None
        self.textip = None
        self.pooledOp = None
        self.input_dfs = None
        self.data_frame = df
        self.feature_df = None
        self.model_name = None
        self.InitModel(model_name, avg_pooling)

    def InitModel(self, model_name, avg_pooling):
      

        if model_name == 'distilBert':
            model_class, tokenizer_class, pretrained_weights = (
                DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
            self.model_name = 'distilBert'

        if model_name == 'mBert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
            self.max_seq_length = 512
            self.model_name = 'mBert'
        
        if model_name == 'mBert_p':
            self.model_name = 'mBert_p'
            self.nlp = pipeline(task ="feature-extraction", model = 'bert-base-multilingual-cased', tokenizer='bert-base-multilingual-cased', framework='pt', device=0)


        if model_name == 'muril':
            self.textip = tf.keras.layers.Input(shape=(), dtype=tf.string)
            self.max_seq_length = 128
            muril_model, muril_layer = self.init_muril(
                model_url="https://tfhub.dev/google/MuRIL/1", max_seq_length=self.max_seq_length,
                avg_pooling=avg_pooling)
            vocab_file = muril_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = muril_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
            self.model_name = 'muril'
            self.model = muril_model
            self.avg_pooling = avg_pooling
        if model_name=='xlmr':    
            self.model_name = 'xlmr'
            self.nlp = pipeline(task ="feature-extraction", model = 'xlm-roberta-base', tokenizer='xlm-roberta-base', framework='pt', device=0)
            self.avg_pooling = avg_pooling 
            
    def tokenize(self, column):
        tokenized_text = column.apply((lambda x: self.tokenizer.encode(x,truncation=True,add_special_tokens=True)))
        max_len = 0
        for i in tokenized_text.values:
            if len(i) > max_len:
                max_len = len(i)
        self.tokenized_padded_text = np.array([i + [0]*(max_len-len(i)) for i in tokenized_text.values])
        self.create_attention_mask()

    def create_attention_mask(self):
        self.attention_mask = np.where(self.tokenized_padded_text != 0, 1, 0)
        print(type(self.tokenized_padded_text))
        self.input_ids = torch.tensor(self.tokenized_padded_text)
        self.attention_mask = torch.tensor(self.attention_mask)

    def init_muril(self, model_url, max_seq_length, avg_pooling):
        inputs = dict(
            input_word_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
            input_mask=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
            input_type_ids=tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32),
        )

        muril_layer = hub.KerasLayer(model_url, trainable=True)
        outputs = muril_layer(inputs)
        print(outputs)
        assert 'sequence_output' in outputs
        assert 'pooled_output' in outputs
        assert 'encoder_outputs' in outputs
        assert 'default' in outputs
        if avg_pooling:
            return tf.keras.Model(inputs=inputs, outputs=outputs["encoder_outputs"]), muril_layer
        else:
            return tf.keras.Model(inputs=inputs, outputs=outputs["pooled_output"]), muril_layer

    def create_input(self, input_strings, tokenizer, max_seq_length):
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

    def encode(self, input_text):
        input_ids, input_mask, input_type_ids = self.create_input(input_text,
                                                                  self.tokenizer,
                                                                  self.max_seq_length)
        inputs = dict(
            input_word_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
        )
        return self.model(inputs)

    def GetFeatures(self, input=None):
        if self.model_name == 'mBert':
            with torch.no_grad():
                last_hidden_states = self.model(self.input_ids, attention_mask=self.attention_mask)
            last_hidden_states['last_hidden_state'].size()
            self.features = last_hidden_states[0][:, 0, :].numpy()
            self.features = pd.DataFrame(self.features)
        elif self.model_name == 'muril':
            embeddings = self.encode(input)
            if not self.avg_pooling:
                self.features = pd.DataFrame(embeddings.numpy())
            else:
                f1 = embeddings[7][:, 0, :].numpy()
                f2 = embeddings[6][:, 0, :].numpy()
                f3 = embeddings[5][:, 0, :].numpy()
                self.features = pd.DataFrame((f1+f2+f3)/3)
        elif self.model_name == 'xlmr':
            sentences=input
            features = self.nlp(sentences, truncation=True) 
            featurelist=list()
            for i in features:
               featurelist.append(i[0][0])
            self.features=pd.DataFrame(featurelist)     
        elif self.model_name == 'mBert_p':
            sentences=input
            features = self.nlp(sentences, truncation=True) 
            featurelist=list()
            for i in features:
               featurelist.append(i[0][0])
            self.features=pd.DataFrame(featurelist)           
        return self.features

# %%
class classifiers:

  def __init__(self,features_train,label_train,features_test,label_test):
    #self.features_set=features
    #self.labels=label
    #self.Createstaticsplit(features,label)
    self.train_features=features_train
    self.train_labels=label_train
    self.test_features=features_test
    self.test_labels=label_test
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
        acc,f1_score,y=self.XGBClassifier(2)
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

  def VotingClassifier(self):
      clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
      clf2 = RandomForestClassifier(n_estimators=50, random_state=1)   
      clf3=XGBClassifier(n_estimators=500,learning_rate=1, max_depth=2,objective='multi:softmax',num_class=2) 
      eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('xgb', clf3)], voting='hard') 
      eclf.fit(self.train_features,self.train_labels)
      y_pred= eclf.predict(self.test_features) 
      return print(accuracy_score(self.test_labels, y_pred)),print(f1_score(self.test_labels, y_pred,average='macro')),y_pred

  def XGBClassifier(self,num_class):
     from xgboost import XGBClassifier
     classifier = XGBClassifier(n_estimators=500,learning_rate=1, max_depth=2,objective='multi:softmax',num_class=num_class)
     classifier.fit(self.train_features, self.train_labels)
     
     y_pred = classifier.predict(self.test_features)
     return print(accuracy_score(self.test_labels, y_pred)),print(f1_score(self.test_labels, y_pred,average='macro')),y_pred

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
     return print(accuracy_score(self.test_labels, y_pred)),print(f1_score(self.test_labels, y_pred,average='macro')),y_pred

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
      #self.labels.replace('NOT',0,inplace=True)
      #self.labels.replace('HOF',1,inplace=True)
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
     # y_pred = (y_pred > 0.5)
      return accuracy_score(test_labels, y_pred),f1_score(test_labels, y_pred,average='macro')

  def LogisticRegression(self):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression 
    lr_clf = LogisticRegression(multi_class='multinomial')
    lr_clf.fit(self.train_features, self.train_labels) 

    y_pred=lr_clf.predict(self.test_features)
    return print(accuracy_score(self.test_labels, y_pred)),print(f1_score(self.test_labels, y_pred,average='macro')),y_pred

# %%
def classify(train_dataframe,test_dataframe,classifier):
   classifier_class=classifiers(features_train=train_dataframe.iloc[:, :-1],label_train=train_dataframe.label,features_test=test_dataframe.iloc[:, :-1],label_test=test_dataframe.label)
   test_labels=test_dataframe.label
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
   elif classifier == 'M_VC': 
      a,f,cm,y_pred=classifier_class.MajorityVotingClassifier(2)  
      print(classification_report(test_labels, y_pred, labels=[0,1])) 
   elif classifier == 'VC': 
      a,f,y_pred=classifier_class.VotingClassifier()  
      print(classification_report(test_labels, y_pred, labels=[0,1])) 

# %%
model_pipeline=model(df,model_name='muril')

# %%
def cosine_similarity(list1, list2, distance=False):
  if distance:
    return spatial.distance.cosine(list1, list2)

  return 1 - spatial.distance.cosine(list1, list2)

# %%
from operator import add
from scipy import spatial


def get_final_embeddings(df):

  row_embeddings = []

  for text, context in zip(df["text"], df["list_context"]):
    
    text_embeddings = model_pipeline.GetFeatures([text]).iloc[0].to_list()

    final_embeddings = np.zeros(768)

    final_embeddings = list( map(add, text_embeddings, final_embeddings) )

    for sub_context in context:

      context_embeddings = model_pipeline.GetFeatures([sub_context]).iloc[0].to_list()

      cosine_distance = cosine_similarity(text_embeddings, context_embeddings, distance=True)

      weighted_contexts = [x * cosine_distance for x in context_embeddings]

      final_embeddings = list( map(add, weighted_contexts, final_embeddings))

    row_embeddings.append(final_embeddings)

  return row_embeddings

# %%
df["final_embeddings"] = get_final_embeddings(df)

# %%
test["final_embeddings"] = get_final_embeddings(test)

# %%
TASK1 = {"HOF": 1, "NONE": 0}

# %%
df['label']=df['label'].map(TASK1)
test['label']=test['label'].map(TASK1)

# %%
train_X = pd.DataFrame(df['final_embeddings'].tolist())
train_y = df["label"]
train_X['label']=df['label']
train_X.to_pickle('muril_p2_train.pkl')

# %%
test_X = pd.DataFrame(test['final_embeddings'].tolist())
test_y = test["label"]
test_X['label']=test['label']
test_X.to_pickle('muril_p2_test.pkl')

# %%
# Classification Head
classify(train_X,test_X,'RF')

# %%
classify(train_X,test_X,'VC')

# %%
classify(train_X,test_X,'LR')

# %%
classify(train_X,test_X,'XGBOOST')

# %%
classify(train_X,test_X,'M_VC')

# %%
'''params_xgboost = {"n_estimators": 500, "learning_rate": 1, "max_depth": 2,
                  "objective": 'multi:softmax', "num_class": 4}
params_lr = {"multi_class": 'multinomial'}
params_rf = {"n_estimators": 500, "criterion": 'entropy', "random_state": 0}

# %%
import pickle
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Classifiers:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.Createstaticsplit(train_X, train_y, test_X, test_y)
        self.accuracy = list()
        self.f1score = list()

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

        with open('xlmr_p2_predictions.pkl', 'wb') as f:
          pickle.dump(y_pred, f)

        return accuracy_score(
            self.test_labels, y_pred), f1_score(
            self.test_labels, y_pred, average='weighted')

    def XGBClassifier(self, num_class):
        classifier = XGBClassifier(**params_xgboost)
        classifier.fit(self.train_features, self.train_labels)
        y_pred = classifier.predict(self.test_features)
        return accuracy_score(self.test_labels, y_pred), f1_score(self.test_labels, y_pred, average='weighted'), y_pred

    def RandomForestClassifier(self):
        classifier = RandomForestClassifier(**params_rf)
        classifier.fit(self.train_features, self.train_labels)
        y_pred = classifier.predict(self.test_features)
        return accuracy_score(self.test_labels, y_pred), f1_score(self.test_labels, y_pred, average='weighted'), y_pred

    def Createstaticsplit(self, train_X, train_y, test_X, test_y):
        self.train_features = train_X
        self.train_labels = train_y
        self.test_features = test_X
        self.test_labels = test_y

    def LogisticRegression(self):
        lr_clf = LogisticRegression(**params_lr)
        lr_clf.fit(self.train_features, self.train_labels)
        y_pred = lr_clf.predict(self.test_features)
        return accuracy_score(self.test_labels, y_pred), f1_score(self.test_labels, y_pred, average='weighted'), y_pred

# %%
handler = Classifiers(train_X, train_y, test_X, test_y)

accuracies,f1_scores = handler.MajorityVotingClassifier(2)

print("Accuracy: {}, F1 Score: {}".format(accuracies, f1_scores))'''

# %%


# %%



