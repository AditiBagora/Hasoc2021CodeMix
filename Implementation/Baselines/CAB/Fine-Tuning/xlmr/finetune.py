import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None) 
train = pd.read_pickle("p2_codemix_flat.pkl")
from transformers import pipeline
class XLMR:
    def __init__(self):
        self.model_name = 'xlmr'
        self.nlp = pipeline(task ="feature-extraction", model = 'google/muril-base-cased', tokenizer='google/muril-base-cased', framework='pt', device=0)

    def GetFeatures(self, sentences=None):
        if self.model_name == 'xlmr':
            features = self.nlp(sentences, truncation=True)      
            return pd.DataFrame(features[0][0])
xlmr = XLMR()
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.linear = nn.Linear(768,2)
          self.outputlayer = nn.LogSoftmax(dim=1)
          self.nlp = pipeline(task ="feature-extraction", model = 'xlm-roberta-base', tokenizer='xlm-roberta-base', framework='pt', device=0)
    def cosine_similarity(self,list1, list2, distance=False):
        if distance:
          return spatial.distance.cosine(list1, list2)
        return 1 - spatial.distance.cosine(list1, list2)

    def concatenate(self, tensor1, tensor2):
      return torch.cat((tensor1, tensor2), 1)
      
    def forward(self, text=None, context=None):
        text_list = self.nlp(text, truncation=True)
        text_embeddings = torch.Tensor(text_list).to(device)
        text_pool = torch.sum(text_embeddings, 1)/text_embeddings.size(1)
        final_embeddings = text_pool.to(device)
        for sub_context in context:
          context_list = self.nlp(sub_context, truncation=True) 
          context_embeddings = torch.Tensor(context_list).to(device)
          context_pool = torch.sum(context_embeddings, 1)/context_embeddings.size(1) 
          cosine_distance = self.cosine_similarity(text_list[0][0], context_list[0][0], distance=True)
          final_embeddings +=  cosine_distance*context_pool
        linear_output = self.linear(final_embeddings)
        outputs = self.outputlayer(linear_output)
        return outputs

model = CustomBERTModel()
learning_rate = 1e-4
epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
model = model.to(device)
from operator import add
from scipy import spatial

print("Training....")

for epoch in range(epochs):
    total_train_loss=0
    print("==========Epochs:{}===========".format(epoch)) 
    row_embeddings=[]

    for idx, (context,text,label) in enumerate(zip(train["list_context"], train["text"], train["label"])):
      optimizer.zero_grad()
      logits=model(text,context) 
 
      loss = criterion(input = logits, target = torch.LongTensor([label]).to(device))
      
      loss.backward()
      optimizer.step()
      total_train_loss += loss
      if idx % 200 == 0:
        print("Loss: {}".format((total_train_loss/len(train)).item()))
    torch.save(model.state_dict(), "rev_xlmr_model_adamw"+str(epoch)+".pth")
    print("Training Loss: {}".format((total_train_loss/len(train)).item()))
torch.save(model.state_dict(), "rev_xlmr_model_adamw.pth")
model = CustomBERTModel()
model.load_state_dict(torch.load("rev_xlmr_model_adamw.pth")) 
test = pd.read_pickle("p2_codemix_flat_test.pkl")

from sklearn.metrics import accuracy_score, f1_score

print("Testing....")

predictions = []
labels = []

model.eval()

model=model.to(device)

with torch.no_grad():
  for  context, text, label in zip(test["list_context"], test["text"], test["label"]):
      logits=model(text,context) 
      y_pred = torch.argmax(logits, dim=-1).item()
      predictions.append(y_pred)
      labels.append(label)
      print([y_pred,label])

  print("Accuracy: {}".format(accuracy_score(predictions, labels)))
  print("F1 Score: {}".format(f1_score(predictions, labels)))
  print(classification_report(predictions, labels, labels=[0,1]))
  import pickle

with open('FT_XLMR_1e-4.pkl', 'wb') as f:
  pickle.dump(predictions, f)                           