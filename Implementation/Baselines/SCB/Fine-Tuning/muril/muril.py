# %%
LEARNING_RATE = 1e-4 
EPSILON = 1e-8
EPOCHS = 4
BATCH_SIZE = 32
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%
import random
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

class Trainer:

    def __init__(self, model_class, tokenizer_class, pretrained_weights, gpu=True, **model_params):

        self.device = self.load_training_device(gpu)

        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.pretrained_weights = pretrained_weights
        self.model = self.load_model(model_class, pretrained_weights, **model_params)
        self.tokenizer = self.load_tokenizer(tokenizer_class, pretrained_weights)

    def load_training_device(self, gpu):
        if gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")

                print('There are %d GPU(s) available.' % torch.cuda.device_count())

                print('We will use the GPU:', torch.cuda.get_device_name(0))
            else:
                print('No GPU available, using the CPU instead.')
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def load_model(self, model_class, pretrained_weights, **model_params):
        model = model_class.from_pretrained(pretrained_weights, **model_params)
        model.to(self.device)
        return model

    def load_tokenizer(self, tokenizer_class, pretrained_weights):
        return tokenizer_class.from_pretrained(pretrained_weights)

    def get_max_tokens(self, sentences):
        max_len = 0
        for sent in sentences:
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        return max_len

    def get_ids_masks(self, sentences, labels, MAX_LEN):
        input_ids = []
        attention_masks = []
        for sentence in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=MAX_LEN,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids.append(encoded_dict['input_ids'])

            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels

    def get_datasets(self, input_ids, attention_masks, labels, validate, train_ratio=0.9):

        dataset = TensorDataset(input_ids, attention_masks, labels)

        if validate:
            train_size = int(train_ratio * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = dataset
            val_dataset = None

        return train_dataset, val_dataset

    def get_dataloader(self, dataset, data_sampler, batch_size=BATCH_SIZE):
        return DataLoader(
            dataset,
            sampler=data_sampler(dataset),
            batch_size=batch_size,
                    )

    def get_scheduler(self, optimizer, training_steps):
        return get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=training_steps)

    def train(self, sentences, labels, epochs, batch_size, optimizer, SEED=42, validate=True, now=0):

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        MAX_LEN = self.get_max_tokens(sentences)

        input_ids, attention_masks, labels = self.get_ids_masks(sentences, labels, 512)

        train_dataset, val_dataset = self.get_datasets(input_ids, attention_masks, labels, validate=validate)

        train_dataloader = self.get_dataloader(train_dataset,RandomSampler, batch_size)
        validation_dataloader = self.get_dataloader(val_dataset, SequentialSampler, batch_size)

        model = self.model
        tokenizer = self.tokenizer

        optimizer = optimizer(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
        scheduler = self.get_scheduler(optimizer, len(train_dataloader)*epochs)

        for epoch in range(epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

            total_train_loss = 0

            model.train()

            for batch in train_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                model.zero_grad()

                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
                loss = result.loss
                logits = result.logits

                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print("Training Loss: {}".format(avg_train_loss))

            if validate:
                print("Running Validation...")
                self.validate(model, validation_dataloader)

            model.save_pretrained("trained_model"+"_"+str(epoch))
            tokenizer.save_pretrained("tokenizer"+"_"+str(epoch))

    def validate(self, model, validation_dataloader):

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                results = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = results.loss
                logits = results.logits

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_loss += loss.item()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("=========================")

# %%
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# %%
import datetime
NOW = datetime.datetime.now()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# %%
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# %%
model_params = {"num_labels":2, "output_attentions":False, "output_hidden_states":False}

# %%
trainer = Trainer(AutoModelForSequenceClassification, AutoTokenizer, 'google/muril-base-cased', **model_params)

# %%
import pandas as pd
train = pd.read_pickle("baseline1_codemix_flat.pkl")

# %%
test= df = pd.read_pickle("baseline1_codemix_flat_test.pkl")

# %%
sentences = train["text"].values
labels = train["label"].values

# %%
lab=list()
for i in labels:
   if i=='HOF':
    lab.append(1)
   else:
    lab.append(0) 

# %%
from transformers import AdamW
OPTIMIZER =AdamW

# %%
trainer.train(sentences, lab, epochs=EPOCHS, batch_size = BATCH_SIZE,optimizer=OPTIMIZER, now=NOW)

# %% [markdown]
# # Testing Pipeline

# %%
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader,SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Tester:
  def __init__(self, model_class, model_file, tokenizer_class, tokenizer_file):
    self.device = torch.device("cpu")
    self.model = self.load_model(model_class, model_file)
    self.tokenizer = self.load_tokenizer(tokenizer_class,tokenizer_file)

  def load_model(self,model_class, model_file):
    model = model_class.from_pretrained(model_file)
    model.to(self.device)
    return model

  def load_tokenizer(self,tokenizer_class, tokenizer_file):
    tokenizer = tokenizer_class.from_pretrained(tokenizer_file)
    return tokenizer

  def get_max_tokens(self,sentences):
    max_len = 0
    for sent in sentences:
        input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    return max_len
  
  def model_inputs(self, sentences, labels, max_len):
    input_ids = []
    attention_masks = []
    for sent in sentences:

      encoded_dict = self.tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = max_len,           
                        pad_to_max_length = True,
                        truncation=True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
   
      input_ids.append(encoded_dict['input_ids'])
  
      attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

  def test(self, sentences, labels):

    max_len = self.get_max_tokens(sentences)

    input_ids, attention_masks, labels = self.model_inputs(sentences, labels, 512)

    batch_size = 32  

    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    self.model.eval()

    results = []

    test_acc = 0.0
    test_f1 = 0.0

    for batch in prediction_dataloader:

      batch = tuple(t.to(self.device) for t in batch)

      b_input_ids, b_input_mask, b_labels = batch

      with torch.no_grad():
          outputs = self.model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

      logits = outputs[0]
      y_pred = torch.argmax(logits, dim = -1)

      test_acc += accuracy_score(b_labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
      test_f1 += f1_score(b_labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average = 'weighted')

    print("Accuracy :{}".format(test_acc/len(prediction_dataloader)))
    print("F1 :{}".format(test_f1/len(prediction_dataloader)))

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tester = Tester(model_class = AutoModelForSequenceClassification, model_file="trained_model_3", tokenizer_class=AutoTokenizer, tokenizer_file="tokenizer_3")

# %%
sentences = test["text"].values
labels = test["label"].values

# %%
lab=list()
for i in labels:
   if i=='HOF':
    lab.append(1)
   else:
    lab.append(0) 

# %%
tester.test(sentences, lab)

# %%



