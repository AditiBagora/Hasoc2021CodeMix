# %%
import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
train = pd.read_pickle("p1_codemix_flat.pkl")

# %%
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.linear = nn.Linear(768,2)
          self.outputlayer = nn.LogSoftmax(dim=1)
          self.nlp = pipeline(task ="feature-extraction", model = 'xlm-roberta-base', tokenizer='xlm-roberta-base', framework='pt', device=0)

    def concatenate(self, tensor1, tensor2):
      return torch.cat((tensor1, tensor2), 1)
      
    def forward(self, contxt=None, text=None):
      text_list = self.nlp(text, truncation=True) 
      context_list=self.nlp(contxt, truncation=True) 
      text_embeddings = torch.FloatTensor(text_list).to(device)
      context_embeddings = torch.FloatTensor(context_list).to(device)
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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
print("Learning rate"+str(learning_rate))
print("Epochs"+str(epochs))
model = model.to(device)

# %%

# %%
lambda1=torch.FloatTensor([0.4]).to(device)
lambda2=torch.FloatTensor([0.6]).to(device)

# %%
print("Training....")
print("Nuber of Examples:", len(train["context"]))
model.train()
for epoch in range(epochs):
    total_train_loss=0
    for idx, (context,text,label) in enumerate(zip(train["context"], train["text"], train["label"])):
      logits=model(context,text)
      loss = criterion(input = logits, target = torch.LongTensor([label]).to(device))
      
      loss.backward()
      optimizer.step()
      total_train_loss += loss
      optimizer.zero_grad()
      if idx %300 == 0:
        print("Current Epoch is : {}/{} || Step is :{}/{}".format(epoch, epochs, idx,  len(train["context"])))
    torch.save(model.state_dict(), "rev_xlmr_model_adamw_"+str(epoch)+".pth")
    print("Training Loss: {}".format((total_train_loss/len(train)).item()))
   
# # %%
torch.save(model.state_dict(), "rev_xlmr_model_adamw_b3.pth")

# %%

model = CustomBERTModel()
#
model.load_state_dict(torch.load("rev_xlmr_model_adamw_b3.pth"))

# %%
# Testing Setup

# %%
test = pd.read_pickle("p1_codemix_flat_test.pkl")
# %%

# %%

# %%

print("Testing....")
predictions = []
labels = []
model.eval()
model=model.to(device)
with torch.no_grad():
  for  context, text, label in zip(test["context"], test["text"], test["label"]):
      logits=model(context,text)  
      y_pred = torch.argmax(logits, dim=-1).item()
      predictions.append(y_pred)
      labels.append(label)

  print("Accuracy: {}".format(accuracy_score(predictions, labels)))
  print("F1 Score: {}".format(f1_score(predictions, labels, average='macro')))
  print(classification_report(predictions, labels, labels=[0,1]))

# %%
import pickle

with open('XLMR_5e-3.pkl', 'wb') as f:
  pickle.dump(predictions, f)


