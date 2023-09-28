from typing import Iterable, List, Optional, Dict
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .base import TrainingModule
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPImageProcessor
from mugen.metrics import Accuracy


class CLIPTrainingModule(TrainingModule):
    def __init__(
        self,
        pretrained_name_or_path: Optional[str] = None,
        clip_config: Optional[Dict] = None,
        tokenizer_config: Optional[Dict] = None,
        image_processor_config: Optional[Dict] = None,
        image_key: str = "image",
        text_key: str = "caption",
    ):
        super().__init__()

        self.image_key = image_key
        self.text_key = text_key

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
            'val/acc': Accuracy(),
        }

    def training_step(self, batch, optimizers: List[Optimizer], batch_idx: int):
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

        logits_per_image = self.model(**model_input).logits_per_image
        preds = logits_per_image.argmax(dim=-1)
        labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)

        self.metrics['val/acc'].update(preds, labels)

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
