import config
import torch


class EntityDataset:

    #passed columns in constructor
    def __init__(self, texts, pos, tags):


        #tokenizing texts
        # texts: [["hi", ",", "my", "name", "is", "nandan"], ["welcome","to","new","topic"]]
        

        #pos .nd ags will be converted into numbers  becoz tags/pos are categorical
        # pos/tags: [[1 2 3 4 1 5], [....] , [....]]


        #in above texts ,pos and tags that are list of lists ==>> each individual list represents each row
        

        self.texts = texts
        self.pos = pos
        self.tags = tags


    #length of dataset
    def __len__(self):
        return len(self.texts)


    #extract item from dataset
    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag =[]

        #text is list of words/tokens 
        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False #no need of CLS and SEP
            )



            #s may be out of vocabulary so it may be splitted in the words that are present in vocabulary ==>> In this way one word can be splitted in multiple words 
            #else that input word (that is s) will remain same as it is.
            #example ==>> a) welcomedearson = welcome ##dear ##son   b) welcome = welcome




            input_len = len(inputs)    #inputs will be list of words 
            ids.extend(inputs)         #extend inputs   #ex. inputs = ["welcome","dear","son"]  ==>> after extending => ids= ["welcome","dear","son"] 
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)




        ids = ids[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]




                       #cls         #sep
        ids =         [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]


        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
