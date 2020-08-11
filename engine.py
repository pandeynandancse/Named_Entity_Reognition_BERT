import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0

    #loop over list of dicts
    for data in tqdm(data_loader, total=len(data_loader)):
        #for each dict
        for k, v in data.items():
            data[k] = v.to(device)

        #when you start your training loop, ideally you should zero out the gradients 
        #so that you do the parameter update correctly. 
        #Else the gradient would point in some other direction than the intended
        # direction towards the minimum (or maximum, in case of maximization objectives).
        optimizer.zero_grad()
        #zero_grad() is restart looping without losses from last step if you use the gradient method for decreasing the error (or losses)
        #if you don't use zero_grad() the loss will be decrease not increase as require
        #for example if you use zero_grad() you will find following output :
        #model training loss is 1.5
        #model training loss is 1.4
        #model training loss is 1.3
        #odel training loss is 1.2
                            #if you don't use zero_grad() you will find following output :
        #model training loss is 1.4
        #model training loss is 1.9
        #model training loss is 2
        ##model training loss is 2.8
        #model training loss is 3.5




        _, _, loss = model(**data)

        #back propagation of loss
        loss.backward()

        #optimizer takes a step towards minimum/maximum as required 
        optimizer.step()

        #change lernning rate via scheduler step
        scheduler.step()

        final_loss += loss.item()
    return final_loss / len(data_loader)





def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
