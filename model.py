"""
Transformer Models
"""
import torch
import torch.nn as nn
from transformers import AutoModel, T5ForConditionalGeneration as T5

from collections import OrderedDict


class Transformer(nn.Module):
    def __init__(self, model_name, num_cls, text2text=False, device_ids=None, num_layers=-1, parallelize=True):
        """
        Transformer model with classification layer (over [CLS])

        :param str model_name: transformer model (e.g. `roberta-base`)
        :param int num_cls: model num of classes (logits layer)
        :param bool text2text: true if text-to-text model (T5)
        :param list[int] device_ids: if set, loads model-parallel version
        """
        super().__init__()
        self.name = model_name
        self.text2text = text2text
        self.softmax = torch.nn.Softmax(dim=1)
        self.parallelized = False
        # T5 Model
        if self.text2text:
            self.model = T5.from_pretrained(model_name)

            # Model Parallel
            if device_ids and len(device_ids) > 1:
                self.parallelized = True
                self.parallelize_t5(device_ids)

        # Others
        else:
            if num_layers > -1:
                self.model = AutoModel.from_pretrained(model_name, num_hidden_layers=num_layers)
            else:
                self.model = AutoModel.from_pretrained(model_name)

            hidden_dim = self.model.config.hidden_size
            self.logit_layer = nn.Linear(hidden_dim, num_cls)

    def forward(self, inp):
        if self.text2text:
            out = self.model(input_ids=inp['input_tokens'],
                             attention_mask=inp['input_attn_mask'],
                             labels=inp['target_tokens'])
            return out

        else:
            # tokens: [B, L], mask: [B, L]
            x = self.model(inp['tokens'],
                           inp['attn_mask'])[0]  # [B, L, D]
            cls_emb = x[:, 0, :]  # [B, D]

            # logits
            logit = self.logit_layer(cls_emb)  # [B, C]

            return logit

    def inference(self, inp) -> list:
        """
        Used for inference-only (Leaderboard)
        :param inp: batch of `tokens` (input_ids) and `attn_mask` (attention_mask)
        :return: predicted labels (0,1)
        """
        # note: does not support text2text models, sample code for bert style models
        logit = self.forward(inp)  # [B, C]
        pred_labels = logit.argmax(dim=1).tolist()  # [B]
        return pred_labels

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def get_embedding_fn(self):
        return self.model.embeddings

    def parallelize_t5(self, device_ids):
        """
        Implements model parallelism for T5. \n
        Given GPU ids, maps model layers to devices.

        The t5-large can be distributed across 2 or 4 devices, \n
        and t5-3b variant is supported only for 4 devices.

        :param list device_ids: GPU ids, num of devices = 2 or 4
        """
        device_map = None
        num_devices = len(device_ids)

        assert num_devices in [2, 4, 8], "supports 2 or 4 or 8 GPUs"
        assert ('t5-large' in self.name or 't5-3b' in self.name), 'model parallelization supports only t5-large & t5-3b'

        # (to-do) Maybe Map to specified `device_ids` for 2 gpus
        if num_devices == 2:
            device_map = {device_ids[0]: list(range(0, 12)),
                          device_ids[1]: list(range(12, 24))}

        elif num_devices == 4:
            device_map = {device_ids[0]: list(range(0, 6)),
                          device_ids[1]: list(range(6, 12)),
                          device_ids[2]: list(range(12, 18)),
                          device_ids[3]: list(range(18, 24))}
        elif num_devices == 8:
            device_map = {device_ids[0]: list(range(0, 3)), device_ids[1]: list(range(3, 6)),
                          device_ids[2]: list(range(6, 9)), device_ids[3]: list(range(9, 12)),
                          device_ids[4]: list(range(12, 15)),
                          device_ids[5]: list(range(15, 18)), device_ids[6]: list(range(18, 21)),
                          device_ids[7]: list(range(21, 24))
                          }
        self.model.parallelize(device_map)

    def load_weights(self, checkpoint):
        state_dict = checkpoint['model_state_dict']

        state_dict_new = OrderedDict()

        for k, v in state_dict.items():
            k_ = k.replace('module.', '')
            state_dict_new[k_] = v

        self.load_state_dict(state_dict_new)
