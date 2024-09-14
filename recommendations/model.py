from transformers import BertModel, BertConfig
import torch
import torch.nn as nn

class BERT4REC(nn.Module):
    def __init__(self, config_path, num_items):
        super(BERT4REC, self).__init__()
        self.config = BertConfig.from_pretrained(config_path)
        self.bert = BertModel.from_pretrained(config_path)
        self.item_embedding = nn.Embedding(num_items, self.config.hidden_size)
        self.linear = nn.Linear(self.config.hidden_size, num_items)

    def forward(self, input_ids, attention_mask, token_type_ids, item_ids):
        # BERT Encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs.last_hidden_state[:, 0, :]  # Take the first token

        # Item Embedding
        item_embedding = self.item_embedding(item_ids)

        # Concatenate and Predict
        concat = torch.cat([last_hidden_state, item_embedding], dim=1)
        output = self.linear(concat)
        return output
