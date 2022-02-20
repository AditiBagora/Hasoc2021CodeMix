# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# %%
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_hub as hub
from bert import bert_tokenization
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
from transformers import pipeline
     

class model_m:
    def __init__(self, df=None, model_name=None, avg_pooling=False):
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
        if model_name == 'muril':
            self.textip = tf.keras.layers.Input(shape=(), dtype=tf.string)
            self.max_seq_length = 500
            muril_model, muril_layer = self.init_muril(
                model_url="https://tfhub.dev/google/MuRIL/1", max_seq_length=self.max_seq_length,
                avg_pooling=avg_pooling)
            vocab_file = muril_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = muril_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
            self.model_name = 'muril'
            self.model = muril_model
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
        if self.model_name == 'muril':
            embeddings = self.encode(input)
            self.features = embeddings.numpy().tolist()
            return self.features[0]
# %%
# %%
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.linear = nn.Linear(768,2)
          self.outputlayer = nn.Softmax(dim=0)
          self.nlp = pipeline(task ="feature-extraction", model = 'google/muril-base-cased', tokenizer='google/muril-base-cased', framework='pt', device=0)
          self.model_muril= model_m(model_name='muril')

    def concatenate(self, tensor1, tensor2):
      return torch.cat((tensor1, tensor2), 1)
      
    def forward(self, contxt=None, text=None, label=None):
      text_list = self.model_muril.GetFeatures([text])
      context_list=self.model_muril.GetFeatures([contxt])
      text_embeddings = torch.Tensor(text_list).to(device)
      context_embeddings = torch.Tensor(context_list).to(device)
      final_embd = lambda1*text_embeddings +  lambda2*context_embeddings
      linear_output = self.linear(final_embd)
      outputs = self.outputlayer(linear_output)
      output_logits=torch.zeros(1,2)
      output_logits[0]=outputs
      return output_logits.to(device)

# %%
model = CustomBERTModel()

# %%
learning_rate = 5e-3
epochs = 3
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

# %%

# %%
from sklearn.metrics import accuracy_score, f1_score,  classification_report
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


