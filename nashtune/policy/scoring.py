"""How a causal language model's answer probabilities are read, written
without torch so it can be tested anywhere.

For each information set the model sees the chat-formatted prompt, ending with
the template's generation prompt (the assistant turn has started). For each
legal action, and each accepted surface form of its word ("call", "Call"),
the answer's token sequence is appended. The probability of an action is the
exact probability that the model's answer starts with one of its surface
forms: the product of next-token probabilities over the form's tokens, summed
over forms. The policy renormalizes these over the legal actions; the
unnormalized total is the "legal mass".

Batching. A form with m tokens needs the model's next-token distribution at
the last m positions of ``prompt + form[:-1]`` (the form's last token is never
fed to the model). Forms whose inputs coincide share one row; when every form
is a single token, each infoset is scored with one row, the prompt itself.
Rows are left-padded so that "the last k positions" line up across a batch,
which lets the model compute logits for only those positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoringPlan:
    rows: list[list[int]]  # model inputs, unique
    row_infoset: np.ndarray  # which infoset (position in the request) each row serves
    entry_row: np.ndarray  # per scored token: its row
    entry_back: np.ndarray  # per scored token: position counted from the row's end (1 = last)
    entry_token: np.ndarray  # per scored token: the token id
    entry_seg: np.ndarray  # per scored token: its segment (one surface form of one action)
    seg_infoset: np.ndarray  # per segment: infoset position in the request
    seg_action: np.ndarray  # per segment: action index
    keep: int  # largest entry_back: how many final positions need logits

    @property
    def num_segments(self) -> int:
        return int(self.seg_infoset.shape[0])


def plan_scoring(prompt_ids: list[list[int]], form_ids: list[list[list[list[int]]]]) -> ScoringPlan:
    """``prompt_ids[j]``: token ids of infoset j's prompt; ``form_ids[j][a][v]``:
    token ids of surface form v of action a."""
    rows: list[list[int]] = []
    row_infoset: list[int] = []
    index: dict[tuple, int] = {}
    e_row, e_back, e_tok, e_seg = [], [], [], []
    s_inf, s_act = [], []
    for j, (prompt, actions) in enumerate(zip(prompt_ids, form_ids)):
        for a, forms in enumerate(actions):
            for form in forms:
                if not form:
                    raise ValueError("an answer form tokenized to nothing")
                seg = len(s_inf)
                s_inf.append(j)
                s_act.append(a)
                key = (j, tuple(prompt) + tuple(form[:-1]))
                row = index.get(key)
                if row is None:
                    row = len(rows)
                    index[key] = row
                    rows.append(list(prompt) + list(form[:-1]))
                    row_infoset.append(j)
                m = len(form)
                for t, tok in enumerate(form):
                    e_row.append(row)
                    e_back.append(m - t)
                    e_tok.append(int(tok))
                    e_seg.append(seg)
    return ScoringPlan(
        rows=rows, row_infoset=np.array(row_infoset, dtype=np.int64),
        entry_row=np.array(e_row, dtype=np.int64), entry_back=np.array(e_back, dtype=np.int64),
        entry_token=np.array(e_tok, dtype=np.int64), entry_seg=np.array(e_seg, dtype=np.int64),
        seg_infoset=np.array(s_inf, dtype=np.int64), seg_action=np.array(s_act, dtype=np.int64),
        keep=int(max(e_back)) if e_back else 1,
    )


def collate_left(rows: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left-pad to a common length. Returns input ids, attention mask and
    position ids (0 at each row's first real token)."""
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, width - len(r):] = r
        mask[i, width - len(r):] = 1
    positions = np.clip(np.cumsum(mask, axis=1) - 1, 0, None)
    return ids, mask, positions


def kept_index(plan: ScoringPlan) -> np.ndarray:
    """Index into the last ``plan.keep`` positions for each scored token."""
    return plan.keep - plan.entry_back


def reduce_segments(entry_logp: np.ndarray, plan: ScoringPlan, n_actions: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
    """From per-token log-probabilities to (renormalized log-policy per infoset,
    legal mass per infoset)."""
    seg = np.zeros(plan.num_segments)
    np.add.at(seg, plan.entry_seg, entry_logp)
    out, mass = [], []
    for j, n in enumerate(n_actions):
        action_logp = np.full(n, -np.inf)
        for a in range(n):
            forms = seg[(plan.seg_infoset == j) & (plan.seg_action == a)]
            action_logp[a] = np.logaddexp.reduce(forms)
        total = np.logaddexp.reduce(action_logp)
        out.append(action_logp - total)
        mass.append(float(np.exp(total)))
    return out, np.array(mass)


def chunk_by_rows(rows_per_item: list[int], max_rows: int) -> list[list[int]]:
    """Split items into consecutive chunks of at most ``max_rows`` rows (an
    item larger than that gets a chunk of its own)."""
    chunks: list[list[int]] = []
    current: list[int] = []
    count = 0
    for i, r in enumerate(rows_per_item):
        if current and count + r > max_rows:
            chunks.append(current)
            current, count = [], 0
        current.append(i)
        count += r
    if current:
        chunks.append(current)
    return chunks


def surface_forms(word: str, mode: str = "both", leading_space: bool = False) -> list[str]:
    """Accepted spellings of an answer word: ``lower`` (just "call") or
    ``both`` ("call" and "Call")."""
    prefix = " " if leading_space else ""
    forms = [prefix + word]
    if mode == "both":
        forms.append(prefix + word[:1].upper() + word[1:])
    elif mode != "lower":
        raise ValueError(f"unknown surface mode {mode!r} (lower or both)")
    return forms
