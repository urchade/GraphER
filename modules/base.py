import argparse
import json
from abc import abstractmethod
from pathlib import Path
from typing import Union, Optional, Dict

import torch
import torch.nn as nn
import yaml
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch.utils.data import DataLoader

from .data_processor import GrapherData
from .evaluator import Evaluator
from .token_splitter import WhitespaceTokenSplitter, MecabKoTokenSplitter, SpaCyTokenSplitter
from .utils import er_decoder, get_relation_with_span


class GrapherBase(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.data_proc = GrapherData(config)

        if not hasattr(config, 'token_splitter'):
            self.token_splitter = WhitespaceTokenSplitter()
        elif self.config.token_splitter == "spacy":
            lang = getattr(config, 'token_splitter_lang', None)
            self.token_splitter = SpaCyTokenSplitter(lang=lang)
        elif self.config.token_splitter == "mecab-ko":
            self.token_splitter = MecabKoTokenSplitter()

    @abstractmethod
    def forward(self, x):
        pass

    def adjust_logits(self, logits, keep):
        """Adjust logits based on the keep tensor."""
        keep = torch.sigmoid(keep)
        keep = (keep > 0.5).unsqueeze(-1).float()
        adjusted_logits = logits + (1 - keep) * -1e9
        return adjusted_logits

    def predict(self, x, threshold=0.5, output_confidence=False):
        """Predict entities and relations."""
        out = self.forward(x, prediction_mode=True)

        # Adjust relation and entity logits
        out["entity_logits"] = self.adjust_logits(out["entity_logits"], out["keep_ent"])
        out["relation_logits"] = self.adjust_logits(out["relation_logits"], out["keep_rel"])

        # Get entities and relations
        entities, relations = er_decoder(x, out["entity_logits"], out["relation_logits"], out["topK_rel_idx"],
                                         out["max_top_k"], out["candidate_spans_idx"], threshold=threshold,
                                         output_confidence=output_confidence, token_splitter=self.token_splitter)
        return entities, relations

    def evaluate(self, test_data, threshold=0.5, batch_size=12, relation_types=None):
        self.eval()
        data_loader = self.create_dataloader(test_data, batch_size=batch_size, relation_types=relation_types,
                                             shuffle=False)
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in data_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            batch_predictions = self.predict(x, threshold)
            all_preds.extend(batch_predictions)
            all_trues.extend(get_relation_with_span(x))
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1

    def create_dataloader(self, data, entity_types=None, **kwargs) -> DataLoader:
        return self.data_proc.create_dataloader(data, entity_types, **kwargs)

    def save_pretrained(
            self,
            save_directory: Union[str, Path],
            *,
            config: Optional[Union[dict, "DataclassInstance"]] = None,
            repo_id: Optional[str] = None,
            push_to_hub: bool = False,
            **push_to_hub_kwargs,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            if isinstance(config, argparse.Namespace):
                config = vars(config)
            (save_directory / "config.json").write_text(json.dumps(config, indent=2))

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if config is not None:  # kwarg for `push_to_hub`
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",
            strict: bool = False,
            **model_kwargs,
    ):

        # 2. Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_file = Path(model_id) / "pytorch_model.bin"
        if not model_file.exists():
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config_file = Path(model_id) / "config.json"
        if not config_file.exists():
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        config = load_config_as_namespace(config_file)
        model = cls(config)
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        model.load_state_dict(state_dict, strict=strict, assign=True)
        model.to(map_location)
        return model

    def to(self, device):
        super().to(device)
        import flair
        flair.device = device
        return self


def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
