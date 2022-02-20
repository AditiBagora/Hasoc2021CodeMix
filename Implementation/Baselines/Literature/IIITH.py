# %%
import pandas as pd
import os
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
torch.cuda.current_device()

# %%
train = pd.read_pickle("normalised_train.pkl")
test = pd.read_pickle("normalised_test.pkl")

# %%
# train.drop('tweet_id', axis=1, inplace=True)
# test.drop('tweet_id', axis=1, inplace=True)

# %%
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(train, test_size=0.01, stratify = train['label'])

train_df.columns = ['text', 'labels']
valid_df.columns = ['text', 'labels']

print(f"Train Shape: {train_df.shape}, Valid Shape: {valid_df.shape}")


# %%
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 2
TEST_BATCH_SIZE = 8
LEARNING_RATE = 1e-05
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', truncation=True)

# %%
from datasets import ClassLabel

labels = ClassLabel(num_classes = 2, names = ['HOF', 'NONE'])

# %%
class HindiData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, isTest = False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len
        self.isTest = isTest
        if not self.isTest:
            self.targets = self.data.labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        if not self.isTest:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(labels.str2int(self.targets[index]), dtype=torch.float)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            }

# %%
train_data=train_df.reset_index(drop=True)
valid_data = valid_df.reset_index(drop=True)

print("TRAIN Dataset: {}".format(train_data.shape))
print("VALID Dataset: {}".format(valid_data.shape))

training_set = HindiData(train_data, tokenizer, MAX_LEN, False)
valid_set = HindiData(valid_data, tokenizer, MAX_LEN, False)

# %%
test_data=test.reset_index(drop=True)
print("TEST Dataset: {}".format(test_data.shape))
testing_set = HindiData(test_data, tokenizer, MAX_LEN, True)

# %%
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(valid_set, **valid_params)

# %%
test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

testing_loader = DataLoader(testing_set, **test_params)

# %%
import torch
from torch import nn

class CustomXLMRModel(torch.nn.Module):
    def __init__(self,num_labels=2):
        super(CustomXLMRModel, self).__init__()
        self.num_labels = num_labels
        self.xlmr = AutoModel.from_pretrained("xlm-roberta-base", num_labels = num_labels)
        # self.xlmr = RobertaModel.from_pretrained("roberta-base")
        
        ### New layers:
        # self.linear_layer = nn.Linear(512, 1)
        # self.conv_layer = torch.nn.Conv1d(768, 1, kernel_size=3)

        # Fully-connected layer and Dropout 
        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(0.7)
        self.sofmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.xlmr(ids, attention_mask=mask, token_type_ids=token_type_ids, output_hidden_states=True)
        # print(outputs[0].size()) 
        # print(outputs[1].size())
        # print(outputs[2].size()) 
        # #last_hidden_state, pooler_output, allhidden = outputs[0], outputs[1], outputs[2]
        
        # feature_vec = []

        # for i, hstate in enumerate(allhidden[1:]):
            
        #     # print("For hidden state:", i+1)
        #     # print("original shape", hstate.size())

        #     hstate = hstate.permute(0, 2, 1)
        #     # print("transposed shape", hstate.size())
        #     # torch.Size([8, 768, 512])

        #     # conv_out = F.relu(self.conv_layer(hstate))
        #     # conv_out = conv_out.to(device)

        #     # # torch.Size([8, 1, 510])

        #     # # print("shape after conv:", conv_out.size())
            
        #     # max_out = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2])
        #     # print("shape after max pool:", max_out.size())

        #     # torch.Size([8, 1, 1])

        #     feature_vec.append(hstate[:,0,:])

        #result = torch.cat(feature_vec, dim=0).sum(dim=0).unsqueeze(0)
        #result = result.to(device)

        

        mlp_hidden = self.fc(outputs[1])
        pre_logits = self.dropout(mlp_hidden)
        logits = self.sofmax_layer(pre_logits)
        return logits

# model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels = 2)
model = CustomXLMRModel()
model = model.to(device)

# %%
torch.cuda.empty_cache()

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.BCELoss()
# loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-5)
# optimizer = torch.optim.Adadelta(model.parameters(),
#                                lr=2e-5,
#                                rho=0.95)

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

# %%
best_val_loss = 0
best_val_acc = 0

models_path = "models"

def train(epoch):

    #------------------------------------------------#
    #                TRAIN BLOCK                     #
    #------------------------------------------------#
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    y_pred = []
    y_true = []
    y_pred_val = []
    y_true_val = []
    outputs = torch.empty(1,2)
    model.train()


    for indx,data in enumerate(tqdm(training_loader, total=len(training_loader))):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)

        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        outputs = model(ids, mask, token_type_ids)

        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
    
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        y_pred.extend(big_idx.tolist())
        y_true.extend(targets.cpu().numpy())

        if indx%10==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            log_dict = {
                "train_loss_steps": loss_step,
                "train_acc_steps": accu_step,
            }

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #------------------------------------------------#
    #                VALID BLOCK                     #
    #------------------------------------------------#

    val_loss = 0
    n_correct_val = 0
    nb_val_steps = 0
    nb_val_examples = 0
    model.eval()

    for _,data in tqdm(enumerate(valid_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)

    
        vloss = loss_function(outputs, targets)
        val_loss += vloss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct_val += calcuate_accuracy(big_idx, targets)

        nb_val_steps += 1
        nb_val_examples+=targets.size(0)
        
        if _%10==0:
            val_loss_step = val_loss/nb_val_steps
            val_accu_step = (n_correct_val*100)/nb_val_examples 
            # print(f"Training Loss per 10 steps: {loss_step}")
            # print(f"Training Accuracy per 10 steps: {accu_step}")
            log_dict = {
                "val_loss_steps": val_loss_step,
                "val_acc_steps": val_accu_step,
            }

        y_pred_val.extend(big_idx.tolist())
        y_true_val.extend(targets.cpu().numpy())


    #------------------------------------------------#
    #                LOG BLOCK                       #
    #------------------------------------------------#

    print(f'The Total Train Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    print(f'The Total Val Accuracy for Epoch {epoch}: {(n_correct_val*100)/nb_val_examples}')
    epoch_loss_val = tr_loss/nb_val_steps
    epoch_accu_val = (n_correct_val*100)/nb_val_examples
    print(f"Val Loss Epoch: {epoch_loss_val}")
    print(f"Val Accuracy Epoch: {epoch_accu_val}")

    log_dict = {
        "Epoch": epoch,
        "Train Loss": epoch_loss,
        "Train Acc": epoch_accu,
        "Valid Loss": epoch_loss_val,
        "Valid Acc": epoch_accu_val
    }

    print("--------------------------------------------------")

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    epoch_accu = accuracy_score(y_true, y_pred)
    print(f"Train Stats -- Accu: {epoch_accu}, Prec: {precision}, Recall: {recall}, F1: {f1}")

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro')
    epoch_accu = accuracy_score(y_true_val, y_pred_val)
    print(f"Val Stats   -- Accu: {epoch_accu}, Prec: {precision}, Recall: {recall}, F1: {f1}")

    print("\n")

    #------------------------------------------------#
    #                SAVE  BLOCK                     #
    #------------------------------------------------#

    global best_val_loss
    global best_val_acc

    if epoch == 0:
        torch.save(model, os.path.join(models_path, 'best_xlmr_hi_model_full_split_taska.pt'))
        best_val_loss = epoch_loss_val
        best_val_acc = epoch_accu_val

    elif epoch_accu_val > best_val_acc:
        best_val_acc = epoch_accu_val
        torch.save(model, os.path.join(models_path, 'best_xlmr_hi_model_full_split_taska.pt'))

    # elif epoch_loss_val < best_val_loss:
    #     best_val_loss = epoch_loss_val
    #     torch.save(model, os.path.join(models_path, 'best_xlmr_hi_model.pt'))

    print("Best loss so far", best_val_loss)
    print("Best Accu so far", best_val_acc)

    return 

from tqdm import tqdm
EPOCHS = 10
for epoch in range(EPOCHS):
    train(epoch)

# %%


# %% [markdown]
# # Run model on test set and prepare submission

# %%
BEST_MODEL_PATH = os.path.join("models", "best_xlmr_hi_model_full_split_taska.pt")

# %%
testing_model = torch.load(BEST_MODEL_PATH)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
testing_model.to(device)
testing_model.eval()

# %%
with torch.no_grad():

    preds = []

    for _,data in tqdm(enumerate(testing_loader, 0)):
        print(_)
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)

        outputs = testing_model(ids, mask, token_type_ids)
        big_val, big_idx = torch.max(outputs.data, dim=1)
        preds += big_idx.cpu().tolist()

# %%
label_dict = {"HOF":0, "NONE":1}
test["label"]=test["label"].map(label_dict)

# %%
labels = test["label"].values

# %%
print(accuracy_score(labels, preds))

# %%
from sklearn.metrics import f1_score
print(f1_score(labels, preds), average = "macro")

# %%



