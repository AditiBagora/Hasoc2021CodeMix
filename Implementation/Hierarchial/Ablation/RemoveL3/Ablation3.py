# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import classification_report
# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# %%
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
import numpy as np
import pandas as pd

# %%
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None) 

# %%
train = pd.read_pickle("train_hierarchial.pkl")

# %%
#train=train.sample(1000)

# %%
post,context, label = train.post.values, train.context, train.label

# %%
from transformers import pipeline

class XLMR:
    def __init__(self):
        self.model_name = 'xlmr'
        self.nlp = pipeline(task ="feature-extraction", model = 'xlm-roberta-base', tokenizer='xlm-roberta-base', framework='pt', device=0)

    def GetFeatures(self, sentences=None):
        if self.model_name == 'xlmr':
            features = self.nlp(sentences, truncation=True) 
     
        return features[0][0]

# %%
xlmr = XLMR()

# %%
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
     
          self.linear1 = nn.Linear(1536, 768)
          self.linear2 = nn.Linear(768, 300)
          self.linear3 = nn.Linear(1068, 2)
          self.linear4 = nn.Linear(768,2)
          self.outputlayer = nn.LogSoftmax(dim=1)

    def concatenate(self, tensor1, tensor2):
      return torch.cat((tensor1, tensor2), 1)
      
    def forward(self, post=None, context=None, text=None, only_post = False, post_context=False, post_context_text=False):

      if only_post:
        input_post_embeddings = post
        
        linear4_output = self.linear4(input_post_embeddings)

        outputs = self.outputlayer(linear4_output)

      if post_context:

        input_post_embeddings = post

        input_context_embeddings = context

        post_context = self.concatenate(input_post_embeddings, input_context_embeddings)

        linear1_output = self.linear1(post_context)

        linear4_output = self.linear4(linear1_output)

        outputs = self.outputlayer(linear4_output)

      if post_context_text:

        input_post_embeddings = post

        input_context_embeddings = context

        input_text_embeddings = text

        context_text = self.concatenate(input_context_embeddings, input_post_embeddings)

        linear1_output = self.linear1(context_text)

        linear2_output = self.linear2(linear1_output)

        post_context_text = self.concatenate(input_text_embeddings, linear2_output)

        linear3_output = self.linear3(post_context_text)

        outputs = self.outputlayer(linear3_output)

      return outputs

# %%
model = CustomBERTModel()

# %%
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-7:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# %%
learning_rate = 5e-3
epochs = 3
print("Learning rate:"+str(learning_rate))
print("Epochs:"+str(epochs))
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# %%
model = model.to(device)

# %%
print("Training....")

for epoch in range(epochs):
    total_train_loss=0

    print("==========Epochs:{}===========".format(epoch)) 

    for post, context,text, label in zip(train["post"], train["context"], train["text"], train["label"]):

      if post!="" and context=="" and text=="":

        input_post=xlmr.GetFeatures(post)
        input_post = torch.FloatTensor([input_post]).to(device)

        logits = model(input_post, only_post=True)
        

      if post!="" and context!="" and text=="":

        input_post=xlmr.GetFeatures(post)
        input_post = torch.FloatTensor([input_post]).to(device)

        input_context=xlmr.GetFeatures(context)
        input_context = torch.FloatTensor([input_context]).to(device)

        logits = model(input_post, input_context,post_context=True)

      if post!="" and context!="" and text!="":

        input_post=xlmr.GetFeatures(post)
        input_post = torch.Tensor([input_post]).to(device)

        input_context=xlmr.GetFeatures(context)
        input_context = torch.Tensor([input_context]).to(device)

        input_text=xlmr.GetFeatures(text)
        input_text = torch.Tensor([input_text]).to(device)

        logits = model(input_post, input_context,input_text, post_context_text=True)
       
      print(logits)
        
      loss = criterion(input = logits, target = torch.LongTensor([label]).to(device))

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      total_train_loss += loss
      
      torch.save(model.state_dict(), "rev_xlmr_model_adamw.pth")
    print("Training Loss: {}".format((total_train_loss/len(train)).item()))

# %%
torch.save(model.state_dict(), "rev_xlmr_model_adamw.pth")

# %%
model = CustomBERTModel()
model.load_state_dict(torch.load("rev_xlmr_model_adamw.pth"))

# %%
# Testing Setup

# %%
test = pd.read_pickle("test_hierarchial.pkl")

# %%
from sklearn.metrics import accuracy_score, f1_score

print("Testing....")

predictions = []
labels = []

model.eval()

model=model.to(device)

with torch.no_grad():
  for post, context, text, label in zip(test["post"], test["context"], test["text"], test["label"]):

    if post!="" and context=="" and text=="":

      input_post=xlmr.GetFeatures(post)
      input_post = torch.FloatTensor([input_post]).to(device)

      logits = model(input_post, only_post=True)
        
    if post!="" and context!="" and text=="":

      input_post=xlmr.GetFeatures(post)
      input_post = torch.FloatTensor([input_post]).to(device)

      input_context=xlmr.GetFeatures(context)
      input_context = torch.FloatTensor([input_context]).to(device)

      logits = model(input_post, input_context,post_context=True)

    if post!="" and context!="" and text!="":

      input_post=xlmr.GetFeatures(post)
      input_post = torch.Tensor([input_post]).to(device)

      input_context=xlmr.GetFeatures(context)
      input_context = torch.Tensor([input_context]).to(device)

      input_text=xlmr.GetFeatures(text)
      input_text = torch.Tensor([input_text]).to(device)

      logits = model(input_post, input_context,input_text, post_context_text=True)

    y_pred = torch.argmax(logits, dim=-1).item()

    predictions.append(y_pred)

    labels.append(label)

  print("Accuracy: {}".format(accuracy_score(predictions, labels)))
  print("F1 Score: {}".format(f1_score(predictions, labels,average='macro')))
  print(classification_report(predictions, labels, labels=[0,1]))
# %%
import pickle

with open('XLMR_3ep.pkl', 'wb') as f:
  pickle.dump(predictions, f)


