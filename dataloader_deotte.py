


#Create dataset functions

class dataset(Dataset):

    # CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
            'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}

    def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                                is_split_into_words=True,
                                #return_offsets_mapping=True, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_len)
        word_ids = encoding.word_ids()  
        
        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:              
                    label_ids.append( labels_to_ids[word_labels[word_idx]] )
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append( labels_to_ids[word_labels[word_idx]] )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids: 
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)
        
        return item

    def __len__(self):
        return self.len