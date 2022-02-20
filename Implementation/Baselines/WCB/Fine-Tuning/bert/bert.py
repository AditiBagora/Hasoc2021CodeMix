# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
train = pd.read_pickle("p1_codemix_flat.pkl")


# %%
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.linear = nn.Linear(768,2)
          self.outputlayer = nn.LogSoftmax(dim=1)
          self.nlp = pipeline(task ="feature-extraction", model = 'bert-base-multilingual-cased', tokenizer='bert-base-multilingual-cased', framework='pt', device=0)

    def concatenate(self, tensor1, tensor2):
      return torch.cat((tensor1, tensor2), 1)
      
    def forward(self, contxt=None, text=None, label=None):
      text_list = self.nlp(text, truncation=True) 
      context_list=self.nlp(contxt, truncation=True) 
      text_embeddings = torch.Tensor(text_list).to(device)
      context_embeddings = torch.Tensor(context_list).to(device)
      text_pool = torch.sum(text_embeddings, 1)/text_embeddings.size(1)
      context_pool = torch.sum(context_embeddings, 1)/context_embeddings.size(1)
      final_embd = lambda1*context_pool +  lambda2*text_pool
      linear_output = self.linear(final_embd)
      outputs = self.outputlayer(linear_output)
      return outputs

# %%
model = CustomBERTModel()

# %%
learning_rate = 5e-3
epochs = 4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# %%
model = model.to(device)

# %%
from operator import add
from scipy import spatial

# %%
lambda1=0.4
lambda2=0.6

# %%
print("Training....")
print("Nuber of Examples:", len(train["context"]))
for epoch in range(epochs):
    total_train_loss=0
    for idx, (context,text,label) in enumerate(zip(train["context"], train["text"], train["label"])):
      optimizer.zero_grad()
      logits=model(context,text,label)
      loss = criterion(input = logits, target = torch.LongTensor([label]).to(device))
      loss.backward()
      optimizer.step()
      total_train_loss += loss
      if idx %300 == 0:
        print("Current Epoch is : {}/{} || Step is :{}/{}".format(epoch, epochs, idx,  len(train["context"])))
    torch.save(model.state_dict(), "rev_xlmr_model_adamw_b3_"+str(epoch)+".pth")
    print("Training Loss: {}".format((total_train_loss/len(train)).item()))

# # %%
torch.save(model.state_dict(), "rev_xlmr_model_adamw_b3.pth")

# %%
model = CustomBERTModel()
model.load_state_dict(torch.load("rev_xlmr_model_adamw_b3.pth"))

# %%
# Testing Setup

# %%
test = pd.read_pickle("p1_codemix_flat_test.pkl")

# %%
from sklearn.metrics import accuracy_score, f1_score, classification_report
print("Testing....")
predictions = []
labels = []
model.eval()
model=model.to(device)
with torch.no_grad():
  for  context, text, label in zip(test["context"], test["text"], test["label"]):
      logits=model(context,text,label)  
      y_pred = torch.argmax(logits, dim=-1).item()
      print("Pred:" +str(y_pred)+"label:"+str(label))
      predictions.append(y_pred)
      labels.append(label)

  print("Accuracy: {}".format(accuracy_score(predictions, labels)))
  print("F1 Score: {}".format(f1_score(predictions, labels,average='macro')))
  print(classification_report(predictions, labels, labels=[0,1]))
# %%
import pickle

with open('XLMR_5e-3.pkl', 'wb') as f:
  pickle.dump(predictions, f)


