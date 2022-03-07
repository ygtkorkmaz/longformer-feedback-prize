from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

def feedback_model(config_data):
    tokenizer = AutoTokenizer.from_pretrained(config_data['model']['transformer_name'], add_prefix_space=True)
    config_model = AutoConfig.from_pretrained(config_data['model']['transformer_name'])
    config_model.num_labels = 15
    transformer_model = AutoModelForTokenClassification.from_pretrained(config_data['model']['transformer_name'], config=config_model)

    return transformer_model, tokenizer, config_model
