"""A Hugging Face causal language model as a complete poker strategy.

For every information set the policy is the model's exact probability of
answering with each legal action word, renormalized over the legal actions
(see ``scoring.py`` for exactly what is computed). Training adds LoRA adapters
(peft) and minimizes the loss of ``nashtune.grpo`` written in torch
(``torch_loss.py``).

Needs torch, transformers and peft. Tested in CI with a tiny randomly
initialized model; the real runs happen on a Mac (docs/mac-and-github.md).
"""

from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

from ..games.game import Game
from ..prompts import Prompt, render
from .base import Extraction, Policy
from .scoring import chunk_by_rows, collate_left, kept_index, plan_scoring, surface_forms

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def require_torch() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise SystemExit(
            f"The hf backend needs torch and transformers ({exc}). "
            "See docs/mac-and-github.md for the install command.") from None


def pick_device(name: str = "auto"):
    import torch

    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, device):
    import torch

    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
             "float64": torch.float64}
    if name and name != "auto":
        if name not in table:
            raise ValueError(f"unknown dtype {name!r} ({', '.join(table)})")
        return table[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return torch.float32  # the robust choice on Apple silicon (mps) and on the CPU


def render_chat(tokenizer, prompt: Prompt) -> tuple[list[int], bool]:
    """Token ids of the prompt up to where the answer starts, and whether the
    tokenizer's chat template was used. "Thinking" is switched off for
    templates that support the switch (others ignore the extra variable).
    Templates that reject a system message get it merged into the user turn."""
    if getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(prompt.messages(), tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        except Exception:  # e.g. "System role not supported"
            merged = [{"role": "user", "content": prompt.text()}]
            text = tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        return list(tokenizer(text, add_special_tokens=False)["input_ids"]), True
    # A base model without a chat template: plain text ending in "Answer:".
    return list(tokenizer(prompt.text() + "\nAnswer:")["input_ids"]), False


def load_model(model_id: str, device: str = "auto", dtype: str = "auto", adapter: str | None = None,
               trust_remote_code: bool = False, echo=print):
    """Load a model and tokenizer once so several evaluations can share them
    (pass them to ``HFPolicy(model=..., tokenizer=...)``)."""
    require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    echo(f"[hf] loading {model_id} ({str(dt).replace('torch.', '')}) on {dev}"
         + (f" with adapter {adapter}" if adapter else "") + " ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, trust_remote_code=trust_remote_code)
    model.to(device=dev, dtype=dt)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.to(dev)
    model.eval()
    return model, tokenizer


def free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache") and torch.backends.mps.is_available():
            mps.empty_cache()
    except Exception:
        pass


def _keep_keyword(model) -> str | None:
    """Name of the forward() argument that limits logits to the last positions."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        params = inspect.signature(base.forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


class HFPolicy(Policy):
    def __init__(self, game: Game, model_id: str, template: str = "A", adapter: str | None = None,
                 device: str = "auto", dtype: str = "auto", batch_rows: int = 16, surface: str = "both",
                 train: bool = False, lora_r: int = 16, lora_alpha: int = 32, lr: float = 1e-4,
                 max_grad_norm: float = 1.0, seed: int = 0, trust_remote_code: bool = False,
                 model=None, tokenizer=None, echo=print) -> None:
        require_torch()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.game = game
        self.model_id = model_id
        self.template = template
        self.adapter = adapter
        self.surface = surface
        self.batch_rows = max(1, int(batch_rows))
        self.lr = lr
        self.lora_r, self.lora_alpha = lora_r, lora_alpha
        self.max_grad_norm = max_grad_norm
        self.trainable = bool(train)
        torch.manual_seed(seed)
        self.device = pick_device(device)
        self.dtype = pick_dtype(dtype, self.device)

        start = time.time()
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if model is None:
            echo(f"[hf] loading {model_id} ({self.dtype}) on {self.device} ...")
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype,
                                                         trust_remote_code=trust_remote_code)
        already_wrapped = hasattr(model, "peft_config")  # a peft model passed in (adapters attached)
        if not already_wrapped:
            model.to(device=self.device, dtype=self.dtype)
            for p in model.parameters():
                p.requires_grad_(False)  # the base model is never trained
            if adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter, is_trainable=self.trainable)
            elif self.trainable:
                from peft import LoraConfig, get_peft_model

                names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
                targets = [t for t in LORA_TARGETS if t in names] or "all-linear"
                config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=targets,
                                    bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, config)
        elif not self.trainable:
            for p in model.parameters():
                p.requires_grad_(False)
        model.to(self.device)
        model.eval()  # no dropout, during training as well: the policy must be deterministic
        self.model = model
        self.tokenizer = tokenizer
        self.params = [p for p in model.parameters() if p.requires_grad]
        if self.trainable:
            if not self.params:
                raise ValueError("no trainable parameters (LoRA adapters were not attached)")
            self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.pad_id = int(pad) if pad is not None else 0
        self._keep_kw = _keep_keyword(model)

        # Tokenize every prompt and answer form once.
        self.prompt_ids: list[list[int]] = []
        self.form_ids: list[list[list[list[int]]]] = []
        chat = True
        for info in game.infosets:
            prompt = render(game, info, template)
            ids, chat = render_chat(tokenizer, prompt)
            self.prompt_ids.append(ids)
            per_action = []
            for word in prompt.actions:
                forms, seen = [], set()
                for form in surface_forms(word, surface, leading_space=not chat):
                    toks = tuple(int(t) for t in tokenizer(form, add_special_tokens=False)["input_ids"])
                    if toks and toks not in seen:
                        seen.add(toks)
                        forms.append(list(toks))
                if not forms:
                    raise ValueError(f"answer word {word!r} tokenizes to nothing")
                per_action.append(forms)
            self.form_ids.append(per_action)
        self.chat_template = chat
        self.rows_per_infoset = [len(plan_scoring([p], [f]).rows) for p, f in zip(self.prompt_ids, self.form_ids)]
        self.single_token_answers = all(len(f) == 1 for acts in self.form_ids for forms in acts for f in forms)
        self.mean_prompt_tokens = float(np.mean([len(p) for p in self.prompt_ids]))
        self.load_seconds = time.time() - start

    # ------------------------------------------------------------- scoring
    def _entry_logp(self, plan):
        import torch

        ids, mask, pos = collate_left(plan.rows, self.pad_id)
        dev = self.device
        kwargs = {"input_ids": torch.as_tensor(ids, device=dev), "attention_mask": torch.as_tensor(mask, device=dev),
                  "position_ids": torch.as_tensor(pos, device=dev), "use_cache": False}
        if self._keep_kw:
            kwargs[self._keep_kw] = plan.keep
        logits = self.model(**kwargs).logits[:, -plan.keep:, :]
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        rows = torch.as_tensor(plan.entry_row, device=dev)
        cols = torch.as_tensor(kept_index(plan), device=dev)
        toks = torch.as_tensor(plan.entry_token, device=dev)
        return logp[rows, cols, toks]

    def _score(self, ids: list[int]):
        """Renormalized log-policy tensors and legal mass for ``ids`` (keeps the
        autograd graph unless called under torch.no_grad)."""
        import torch

        logs: list = [None] * len(ids)
        mass = [0.0] * len(ids)
        for chunk in chunk_by_rows([self.rows_per_infoset[i] for i in ids], self.batch_rows):
            sub = [ids[k] for k in chunk]
            plan = plan_scoring([self.prompt_ids[i] for i in sub], [self.form_ids[i] for i in sub])
            entry = self._entry_logp(plan)
            seg_index = torch.as_tensor(plan.entry_seg, device=entry.device)
            seg = torch.zeros(plan.num_segments, dtype=entry.dtype, device=entry.device).index_add(0, seg_index, entry)
            for local, k in enumerate(chunk):
                n = self.game.infosets[ids[k]].action_count
                per_action = []
                for a in range(n):
                    where = np.flatnonzero((plan.seg_infoset == local) & (plan.seg_action == a))
                    per_action.append(torch.logsumexp(seg[torch.as_tensor(where, device=entry.device)], dim=0))
                action_logp = torch.stack(per_action)
                total = torch.logsumexp(action_logp, dim=0)
                logs[k] = action_logp - total
                mass[k] = float(torch.exp(total.detach()).cpu())
        return logs, mass

    def logpi(self, ids) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            logs, _ = self._score([int(i) for i in ids])
        return [lp.detach().cpu().double().numpy() for lp in logs]

    def extract(self) -> Extraction:
        import torch

        start = time.time()
        ids = list(range(self.game.num_infosets))
        with torch.no_grad():
            logs, mass = self._score(ids)
        strategy = np.empty(self.game.action_slots)
        for info, lp in zip(self.game.infosets, logs):
            p = np.exp(lp.detach().cpu().double().numpy())
            strategy[info.offset : info.offset + info.action_count] = p / p.sum()
        if not np.all(np.isfinite(strategy)):
            raise RuntimeError(f"the model produced non-finite probabilities in {self.dtype}; "
                               "rerun with --dtype float32")
        return Extraction(strategy=strategy, legal_mass=np.array(mass), seconds=time.time() - start,
                          extra={"single_token_answers": self.single_token_answers,
                                 "mean_prompt_tokens": self.mean_prompt_tokens})

    # ------------------------------------------------------------ training
    def update(self, batch, n_updates: int = 1) -> dict:
        import torch

        from .torch_loss import loss_from_logp_torch

        if not self.trainable:
            raise RuntimeError("this policy was loaded without train=True")
        if len(batch) == 0:
            return {"loss": 0.0, "grad_norm": 0.0}  # nothing contributes; AdamW would skip every parameter
        chunks = chunk_by_rows([self.rows_per_infoset[int(i)] for i in batch.ids], self.batch_rows)
        first_loss, grad_norm = None, float("nan")
        for _ in range(n_updates):
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for chunk in chunks:  # gradient accumulation: the loss is a sum over infosets
                sub = batch.subset(chunk)
                logs, _ = self._score([int(i) for i in sub.ids])
                loss = loss_from_logp_torch(logs, sub)
                loss.backward()
                total += float(loss.detach().cpu())
            if self.max_grad_norm:
                norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            else:
                norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in self.params if p.grad is not None))
            self.optimizer.step()
            if first_loss is None:
                first_loss, grad_norm = total, float(norm.detach().cpu())
        return {"loss": first_loss, "grad_norm": grad_norm}

    # --------------------------------------------------------- persistence
    def save_state(self, directory) -> None:
        import torch

        directory = Path(directory)
        state = {
            "trainable": {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad},
            "optimizer": self.optimizer.state_dict() if self.trainable else None,
        }
        torch.save(state, directory / "hf_state.pt")
        (directory / "hf_meta.json").write_text(json.dumps(self.describe(), indent=1) + "\n")

    def load_state(self, directory) -> None:
        import torch

        state = torch.load(Path(directory) / "hf_state.pt", map_location="cpu", weights_only=False)
        named = dict(self.model.named_parameters())
        unknown = [n for n in state["trainable"] if n not in named]
        if unknown:
            raise ValueError(f"checkpoint does not fit this model (e.g. {unknown[0]})")
        with torch.no_grad():
            for n, tensor in state["trainable"].items():
                named[n].copy_(tensor.to(device=named[n].device, dtype=named[n].dtype))
        if self.trainable and state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])

    def save(self, directory) -> None:
        """Write the LoRA adapter (loadable with ``--adapter``)."""
        if not self.trainable or not hasattr(self.model, "save_pretrained"):
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory / "adapter"))
        (directory / "adapter_info.json").write_text(json.dumps(self.describe(), indent=1) + "\n")

    def describe(self) -> dict:
        import torch
        import transformers

        info = {
            "backend": "hf", "model": self.model_id, "adapter": self.adapter, "template": self.template,
            "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""), "surface": self.surface,
            "chat_template": self.chat_template, "single_token_answers": self.single_token_answers,
            "mean_prompt_tokens": round(self.mean_prompt_tokens, 1), "batch_rows": self.batch_rows,
            "torch": torch.__version__, "transformers": transformers.__version__,
        }
        if self.trainable:
            info.update({"lora_r": self.lora_r, "lora_alpha": self.lora_alpha, "lr": self.lr,
                         "max_grad_norm": self.max_grad_norm,
                         "trainable_parameters": int(sum(p.numel() for p in self.params))})
        try:
            import peft

            info["peft"] = peft.__version__
        except ImportError:
            pass
        return info


def bench(policy: HFPolicy, n_infosets: int = 32, group_size: int = 8, echo=print) -> dict:
    """Measure seconds per scored infoset and per training step on this machine."""
    from ..grpo import BatchBuilder
    from ..rewards import refresh

    game = policy.game
    rng = np.random.default_rng(0)
    ids = list(range(min(n_infosets, game.num_infosets)))
    start = time.time()
    logs = policy.logpi(ids)
    score_s = (time.time() - start) / len(ids)
    out = {"seconds_per_infoset_scored": score_s,
           "seconds_full_extraction_estimate": score_s * game.num_infosets,
           "mean_prompt_tokens": policy.mean_prompt_tokens, "single_token_answers": policy.single_token_answers}
    if policy.trainable:
        info = refresh(game, game.uniform_strategy())
        builder = BatchBuilder([i.action_count for i in game.infosets])
        for i, lp in zip(ids, logs):
            info_slots = slice(game.infosets[i].offset, game.infosets[i].offset + game.infosets[i].action_count)
            builder.set_old(i, lp)
            p = np.exp(lp) / np.exp(lp).sum()
            actions = rng.choice(len(p), size=group_size, p=p)
            r = info.q[info_slots][actions] + rng.normal(scale=1e-3, size=group_size)
            adv = (r - r.mean()) / (r.std() + 1e-6)
            for a, A in zip(actions, adv):
                builder.add_sample(i, int(a), float(A), 1.0 / (len(ids) * group_size))
        batch = builder.build()
        import copy

        state = {n: p.detach().clone() for n, p in policy.model.named_parameters() if p.requires_grad}
        opt_state = copy.deepcopy(policy.optimizer.state_dict())
        start = time.time()
        policy.update(batch, 1)
        out["seconds_per_train_step"] = time.time() - start
        out["train_infosets"] = len(ids)
        import torch

        with torch.no_grad():  # undo the benchmark step
            named = dict(policy.model.named_parameters())
            for n, t in state.items():
                named[n].copy_(t)
        policy.optimizer.load_state_dict(opt_state)
    if math.isfinite(score_s):
        echo(json.dumps(out, indent=1))
    return out
