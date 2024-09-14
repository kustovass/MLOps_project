import torch
from transformers import BertTokenizer
import yaml

from recommendations.model import BERT4REC

def infer(user_history, item_id, model_path="bert4rec_model.pth", config_path="config.yaml"):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model_save"]
    device = torch.device("cuda" if config["device"]["use_cuda"] else "cpu")


    model = BERT4REC(config_path=config['model']['name'], num_items=config['model']['num_items']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])


    encoded_input = tokenizer(user_history, return_tensors='pt', padding='max_length', truncation=True)

    with torch.no_grad():
        output = model(
            encoded_input['input_ids'].to(device),
            encoded_input['attention_mask'].to(device),
            encoded_input['token_type_ids'].to(device),
            torch.tensor([item_id]).to(device)
        )


    return torch.nn.functional.softmax(output, dim=1)[0, item_id].item()


