from typing import Iterable, List, Optional, Dict
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .base import TrainingModule
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPImageProcessor
from mugen.metrics import MeanMetric


class CLIPTrainingModule(TrainingModule):
    LORA_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]

    def __init__(
        self,
        pretrained_name_or_path: Optional[str] = None,
        clip_config: Optional[Dict] = None,
        tokenizer_config: Optional[Dict] = None,
        image_processor_config: Optional[Dict] = None,
        image_key: str = "image",
        text_key: str = "caption",
        random_truncation: bool = False,
    ):
        super().__init__()

        self.image_key = image_key
        self.text_key = text_key
        self.random_truncation = random_truncation

        assert (
            clip_config is not None or pretrained_name_or_path is not None
        ), "Either clip_config or pretrained_name_or_path must be specified!"
        if pretrained_name_or_path is not None:
            self.model = CLIPModel.from_pretrained(pretrained_name_or_path)
            self.processor = CLIPProcessor.from_pretrained(pretrained_name_or_path)
        
        if clip_config is not None:
            self.model = CLIPModel(**clip_config)
        
        if tokenizer_config is not None and image_processor_config is not None:
            tokenizer = CLIPTokenizer(**tokenizer_config)
            # image_processor_config.update({'do_rescale': False})
            image_processor = CLIPImageProcessor(**image_processor_config)
            self.processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)

        if self.processor.image_processor.do_rescale:
            print('Feature extractor should not rescale images. Setting do_rescale to False')
            self.processor.image_processor.do_rescale = False

        self.metrics = {
            'val/cosine_similarity': MeanMetric(),
        }

    def training_step(self, batch, optimizers: List[Optimizer], batch_idx: int):
        imgs = batch[self.image_key]
        texts = batch[self.text_key]
        model_max_length = self.processor.tokenizer.model_max_length

        processor_args = dict(text=texts, images=imgs, return_tensors='pt')

        if self.random_truncation:
            processor_args.update(padding=True, return_length=True)
        else:
            processor_args.update(padding='max_length', max_length=model_max_length, truncation=True)

        model_input = self.processor(**processor_args)

        if self.random_truncation:
            length = model_input.pop('length')
            truncated_input_ids = torch.zeros((len(length), model_max_length), dtype=torch.long)
            truncated_attention_mask = torch.zeros((len(length), model_max_length), dtype=torch.long)

            for i in range(len(length)):
                if length[i] > model_max_length:
                    trunc_start_idx = torch.randint(0, length[i] - model_max_length, (1,)).item()
                else:
                    trunc_start_idx = 0

                truncated_input_ids[i] = model_input['input_ids'][i][trunc_start_idx: trunc_start_idx + model_max_length]
                truncated_attention_mask[i] = model_input['attention_mask'][i][trunc_start_idx: trunc_start_idx + model_max_length]
            model_input['input_ids'] = truncated_input_ids
            model_input['attention_mask'] = truncated_attention_mask

        for k in model_input:
            model_input[k] = model_input[k].to(self.device)

        model_output = self.model(**model_input, return_loss=True)
        self.backward_loss(model_output.loss)

        opt = optimizers[0]
        opt.step()
        opt.zero_grad()

        self.log({"train/loss": model_output.loss.item()})

    def validation_step(self, batch, batch_idx: int):
        imgs = batch[self.image_key]
        texts = batch[self.text_key]

        model_input = self.processor(
            text=texts,
            images=imgs,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        for k in model_input:
            model_input[k] = model_input[k].to(self.device)

        model_out = self.model(**model_input)
        similarity = F.cosine_similarity(model_out.text_embeds, model_out.image_embeds, dim=-1)

        self.metrics['val/cosine_similarity'].update(similarity.mean().item())

    def on_validation_epoch_end(self):
        self.log({k: metric.compute() for k, metric in self.metrics.items()})
        for metric in self.metrics.values():
            metric.reset()

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        return [self.model.parameters()]

    def save_model_hook(self, models, weights, output_dir):
        for i, model in enumerate(models):
            model.model.save_pretrained(osp.join(output_dir, "text_encoder"))
            weights.pop()

        self.processor.tokenizer.save_pretrained(osp.join(output_dir, "tokenizer"))

    def load_model_hook(self, models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = CLIPModel.from_pretrained(osp.join(input_dir, "text_encoder"))
            # model.model.register_to_config(**load_model.config)

            model.model.load_state_dict(load_model.state_dict())
            del load_model

        self.processor.tokenizer = CLIPTokenizer.from_pretrained(osp.join(input_dir, "tokenizer"))
