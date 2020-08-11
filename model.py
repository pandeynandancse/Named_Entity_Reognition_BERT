import config
import torch
import transformers
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):



    #classification problem so cross entropy
    lfn = nn.CrossEntropyLoss()



    #concept of active loss has been borrowed from hugging face
    # you don't need to calculate the loss for whole sentence -- >>> you just need to calcuate loss where you don't have padding that means where is 1 there you have to calculate loss
    active_loss = mask.view(-1) == 1
    
    active_logits = output.view(-1, num_labels)
    

    #if active loss is false then replace it  with ==>>torch.tensor(lfn.ignore_index).type_as(target)  ===>>>  value of lfn.ignore_index is -100
    #SO simply ignore the index where value is -100 for calculating loss
    active_labels = torch.where(
        active_loss,
        target.view(-1),  #active_loss and target.view(-1) will have same length  
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)   #here lfn will automatically ignore the index where vlaue = -100 and in this way --->>> you calcuate loss where you don't have padding that means where is 1 there you have to calculate loss
    return loss




class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)


        
        #to reduce overfitting --->> regularization technique -->> dropout --->> given percentage of cells are destroyed
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)



        #768 is number of outputs of BERT uncased
        #linear layer applies liner transforation :  y = xA^T + b
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)
    
    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        #o1 is output1  i.e. sequence output and _ is output2 and here we are taking sequence output into consideration
        # ===>>>>  becoz you are not prediciting one value but one value for each token  ==>> so use sequence output i.e.o1 here
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)


        #becoz multilabel problem as two target columns are present one is pos and another is tag ===>>> so one output is passed parallely into two layers
        bo_tag = self.bert_drop_1(o1)
        bo_pos = self.bert_drop_2(o1)



        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)



        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss
