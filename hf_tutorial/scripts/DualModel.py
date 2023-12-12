from transformers import BertPreTrainedModel, PretrainedConfig, BertModel
from typing import Optional, Tuple, Union
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
import torch

class DualModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: get sentence 1 and sentence 2 inputs
        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]
        senA_attention_mask, senB_attention_mask = attention_mask[:, 0], attention_mask[:, 1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:, 0], token_type_ids[:, 1]

        # step 2: get sentence 1 and sentence 2 embeddings
        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [cls] token embedding
        # [batch_size, hidden_size]
        senA_pooled_output = senA_outputs[1]        

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [cls] token embedding
        # [batch_size, hidden_size]
        senB_pooled_output = senB_outputs[1]  

        # step 3: calculate similarity
        # [batch_size]
        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)
        
        # step 4: calculate loss
        loss = None
        if labels is not None:
            loss_fct = CosineEmbeddingLoss(margin=0.3)
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels.float())
        
        # step 5: return
        output = (cos,)
        return ((loss,) + output) if loss is not None else output  
