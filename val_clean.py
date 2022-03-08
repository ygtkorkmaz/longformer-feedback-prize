


def __val(df=val_dataset, loader=val_loader):
    
    # put model in training mode
    self.__model.eval()
    
    #PT1: Get predicted labels for all data
    pred_label = []
    
    for batch in loader:

        # MOVE BATCH TO GPU AND INFER
        ids = batch["input_ids"].to(self.__device)
        mask = batch["attention_mask"].to(self.__device)
        outputs = self.__model(ids, attention_mask=mask, return_dict=False)
        all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() #Gives most likely class
        
        # Get prediction for each instance in batch
        predictions = []
        for k,text_preds in enumerate(all_preds):
            token_preds = [id_target_map[i] for i in text_preds]

            #For each set of predictions in the batch, return the prediction of the first token of ea word
            prediction = []
            word_ids = batch['wids'][k].numpy()
            previous_word_idx = -1
            for idx,word_idx in enumerate(word_ids):                            
                if word_idx == -1:
                    pass
                elif word_idx != previous_word_idx:              
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx
            predictions.append(prediction)

        pred_label.extend(predictions)

    #PT2: Take predictons and output df that has string of word ids for each class (lead, claim, etc.)

    final_prediction = []
    
    for i in range(len(df)):

        idx = df.id.values[i]
        pred = pred_label[i] # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1
            else: cls = cls.replace('B','I') # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            
            if cls != 'O' and cls != '' and end - j > 7:
                final_prediction.append((idx, cls.replace('I-',''),
                                     ' '.join(map(str, list(range(j, end))))))
        
            j = end
        
    oof = pd.DataFrame(final_prediction)
    oof.columns = ['id','class','predictionstring']

    #PT3: Finally, line up ground truth and predictions and calculate overlap
    f1s = []
    CLASSES = oof['class'].unique()
    print()
    for c in CLASSES:
        pred_df = oof.loc[oof['class']==c].copy()

        #How are we setting up gt?
        gt_df = df.loc[df['discourse_type']==c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(c,f1)
        f1s.append(f1)
    print()
    print('Overall',np.mean(f1s))
    print()
    
    return f1s, np.mean(f1s)