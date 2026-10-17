"""Deterministic natural-language rendering of an information set.

A prompt is a short rules description (the system message) plus the situation
(seat, private card, public card if dealt, the betting so far narrated round by
round with amounts, the pot, the amount to call) and the legal actions, with an
instruction to answer in one word.

Two independent templates:

* ``A`` is used for training.
* ``B`` is held out: different wording, card names written as letters, and the
  parts in a different order (actions first, then history, then cards). It
  tests whether what a model learned from template A survives a paraphrase.

Prompts are kept short because scoring them is the main compute cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .games.game import Game, Infoset
from .games.poker import RANK_WORDS, PokerRules

ACTION_WORDS = {"k": "check", "b": "bet", "f": "fold", "c": "call", "r": "raise"}
WORD_ACTIONS = {word: letter for letter, word in ACTION_WORDS.items()}
TEMPLATES = ("A", "B")


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    actions: tuple[str, ...]  # legal action words, in the infoset's action order

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user}]

    def text(self) -> str:
        return f"{self.system}\n\n{self.user}"


@dataclass(frozen=True)
class _Step:
    actor: int
    action: str
    amount: int  # chips put in by this action


@dataclass(frozen=True)
class Situation:
    me: int
    rounds: list[list[_Step]]  # steps per betting round so far
    pot: int
    to_call: int
    round: int


def replay(rules: PokerRules, history: str, me: int) -> Situation:
    """Replay a betting history to recover amounts, the pot and the amount owed."""
    paid = [rules.ante, rules.ante]
    rounds: list[list[_Step]] = [[]]
    rnd, actor = 0, 0
    for ch in history:
        if ch == "/":
            rnd += 1
            actor = 0
            rounds.append([])
            continue
        other = 1 - actor
        owed = paid[other] - paid[actor]
        if ch == "k":
            amount = 0
        elif ch == "c":
            amount = owed
        elif ch in "br":
            amount = owed + rules.bet[rnd]
        else:
            raise ValueError(f"unexpected action {ch!r} in history {history!r}")
        paid[actor] += amount
        rounds[-1].append(_Step(actor, ch, amount))
        actor = other
    if actor != me:
        raise ValueError(f"history {history!r} does not put player {me} to act")
    return Situation(me=me, rounds=rounds, pot=paid[0] + paid[1], to_call=paid[1 - me] - paid[me], round=rnd)


def _join(words: list[str], last: str = "and") -> str:
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + f" {last} " + words[-1]


def _number(n: int) -> str:
    return ["zero", "one", "two", "three", "four"][n] if n < 5 else str(n)


def _plural(word: str) -> str:
    return word + "es" if word.endswith("x") else word + "s"


def _chips(n: int) -> str:
    return f"{n} chip" if n == 1 else f"{n} chips"


def _letter(rank: str) -> str:
    return "10" if rank == "T" else rank


def _game_title(game: Game) -> str:
    if game.name == "kuhn":
        return "Kuhn poker"
    return "Leduc poker" if game.name == "leduc" else f"Leduc poker with {game.rules.ranks} ranks"


# ---------------------------------------------------------------- template A
def _system_a(game: Game) -> str:
    r: PokerRules = game.rules
    words = [RANK_WORDS[x] for x in game.rank_names]
    n = r.ranks * r.copies
    if r.copies == 1:
        deck = f"The deck has {_number(n) if n < 5 else n} cards, from lowest to highest: {_join(words)}."
    else:
        deck = f"The deck has {n} cards, {_number(r.copies)} of each rank, from lowest to highest: {_join(words)}."
    parts = [f"You are playing {_game_title(game)} against one opponent.", deck,
             f"Each player antes {r.ante} chip and is dealt one private card."]
    if r.rounds == 1:
        parts.append("There is one betting round, and the first player acts first.")
    else:
        parts.append("There are two betting rounds, and one public card is revealed after the first. "
                     "The first player acts first in each round.")
    if r.rounds == 1:
        size = f"A bet is {r.bet[0]} chip" + ("s" if r.bet[0] > 1 else "")
    else:
        size = f"A bet or raise is {r.bet[0]} chips in round 1 and {r.bet[1]} chips in round 2"
    if r.max_raises == 1:
        parts.append(f"{size}, and at most one bet is allowed per round.")
    else:
        parts.append(f"{size}, and each round allows at most {_number(r.max_raises)} bets (a bet and a raise).")
    if r.rounds == 1:
        parts.append("If nobody folds, the higher card wins the pot.")
    else:
        parts.append("If nobody folds, a card that pairs the public card wins; otherwise the higher card wins; "
                     "equal cards split the pot.")
    parts.append("Answer with exactly one word.")
    return " ".join(parts)


_PAST_A = {"k": "checked", "b": "bet", "c": "called", "r": "raised"}


def _user_a(game: Game, info: Infoset, sit: Situation) -> str:
    r: PokerRules = game.rules
    seat = "first" if info.player == 0 else "second"
    lines = [f"You are the {seat} player. Your card: {RANK_WORDS[game.rank_names[info.private_rank]]}."]
    if r.rounds > 1:
        public = "not dealt yet" if info.public_rank < 0 else RANK_WORDS[game.rank_names[info.public_rank]]
        lines[0] += f" Public card: {public}."
    for k, steps in enumerate(sit.rounds):
        label = "Betting" if r.rounds == 1 else f"Round {k + 1}"
        lines.append(f"{label}: {_narrate_steps_a(steps, info.player, r, k) if steps else 'no actions yet.'}")
    lines.append(f"Pot: {_chips(sit.pot)}. To call: {_chips(sit.to_call)}.")
    lines.append(f"Legal actions: {', '.join(ACTION_WORDS[a] for a in info.actions)}.")
    lines.append("Your action?")
    return "\n".join(lines)


def _narrate_steps_a(steps: list[_Step], me: int, rules: PokerRules, rnd: int) -> str:
    out = []
    for s in steps:
        who = "You" if s.actor == me else "Opponent"
        if s.action == "b":
            out.append(f"{who} bet {s.amount}.")
        elif s.action == "r":
            out.append(f"{who} raised {rules.bet[rnd]}.")  # called the bet and added one more
        else:
            out.append(f"{who} {_PAST_A[s.action]}.")
    return " ".join(out)


# ---------------------------------------------------------------- template B
def _system_b(game: Game) -> str:
    r: PokerRules = game.rules
    letters = [_letter(x) for x in game.rank_names]
    kind = "one of each" if r.copies == 1 else f"{_number(r.copies)} of each"
    parts = [f"Game: {_game_title(game)}, heads-up.",
             f"Cards: {', '.join(letters)} ({kind}), ranked low to high.",
             f"Everyone posts {r.ante} chip before the deal and receives one hidden card."]
    if r.rounds == 1:
        parts.append("There is a single betting round and the first seat speaks first.")
        parts.append(f"Bet size: {r.bet[0]} chip; only one bet is allowed.")
        parts.append("If nobody folds, the higher card wins.")
    else:
        parts.append("Betting happens twice, and a shared board card is turned face up between the two rounds. "
                     "The first seat speaks first in both rounds.")
        parts.append(f"Bet size: {r.bet[0]} chips before the board card, {r.bet[1]} after; "
                     f"a round may contain at most {_number(r.max_raises)} bets (one bet and one raise).")
        parts.append("At the end, pairing the board card wins; without a pair the higher card wins; identical ranks split.")
    parts.append("Reply with a single word.")
    return " ".join(parts)


_PRESENT_B = {"k": "check", "b": "bet", "c": "call", "r": "raise"}
_PRESENT_B_THIRD = {"k": "checks", "b": "bets", "c": "calls", "r": "raises"}


def _narrate_b(steps: list[_Step], me: int, rules: PokerRules, rnd: int) -> str:
    out = []
    for s in steps:
        if s.actor == me:
            verb = _PRESENT_B[s.action]
            subject = "you"
        else:
            verb = _PRESENT_B_THIRD[s.action]
            subject = "your opponent"
        if s.action == "b":
            out.append(f"{subject} {verb} {s.amount}")
        elif s.action == "r":
            out.append(f"{subject} {verb} by {rules.bet[rnd]}")
        else:
            out.append(f"{subject} {verb}")
    return ", ".join(out)


def _user_b(game: Game, info: Infoset, sit: Situation) -> str:
    r: PokerRules = game.rules
    options = [ACTION_WORDS[a] for a in reversed(info.actions)]
    lines = [f"Choose one of: {', '.join(options)}."]
    if r.rounds == 1:
        story = _narrate_b(sit.rounds[0], info.player, r, 0) if sit.rounds[0] else "no one has acted"
        lines.append(f"So far: {story}.")
    else:
        pieces = []
        for k, steps in enumerate(sit.rounds):
            when = "before the board card" if k == 0 else "after the board card"
            pieces.append(f"{when}, {_narrate_b(steps, info.player, r, k) if steps else 'no one has acted yet'}")
        story = "; ".join(pieces)
        lines.append(f"So far: {story}.")
    card = _letter(game.rank_names[info.private_rank])
    seat = "first" if info.player == 0 else "second"
    if r.rounds == 1 or info.public_rank < 0:
        extra = "" if r.rounds == 1 else "; the board card is not out yet"
        lines.append(f"You hold the {card}{extra}. You are in the {seat} seat.")
    else:
        lines.append(f"You hold the {card}; the board card is the {_letter(game.rank_names[info.public_rank])}. "
                     f"You are in the {seat} seat.")
    if sit.to_call:
        lines.append(f"There are {sit.pot} chips in the middle, and staying in costs {sit.to_call}.")
    else:
        lines.append(f"There are {sit.pot} chips in the middle, and nothing to call.")
    return "\n".join(lines)


# ------------------------------------------------------------------- public
def render(game: Game, info: Infoset | int | str, template: str = "A", answer: str = "word") -> Prompt:
    """The prompt for one infoset. ``answer="json"`` asks for a mixed strategy
    as a JSON object instead of one word (the OpenAI ``stated`` mode)."""
    if isinstance(info, int):
        info = game.infosets[info]
    elif isinstance(info, str):
        info = game.infoset_by_key(info)
    sit = replay(game.rules, info.history, info.player)
    actions = tuple(ACTION_WORDS[a] for a in info.actions)
    if template == "A":
        prompt = Prompt(_system_a(game), _user_a(game, info, sit), actions)
    elif template == "B":
        prompt = Prompt(_system_b(game), _user_b(game, info, sit), actions)
    else:
        raise ValueError(f"unknown template {template!r} (A or B)")
    if answer == "word":
        return prompt
    if answer != "json":
        raise ValueError(f"unknown answer format {answer!r} (word or json)")
    system = prompt.system.replace(_ONE_WORD[template], _JSON_SYSTEM)
    user = prompt.user
    if template == "A":
        user = user.replace("Your action?", _JSON_USER)
    else:
        user = user + "\n" + _JSON_USER
    return Prompt(system, user, actions)


_ONE_WORD = {"A": "Answer with exactly one word.", "B": "Reply with a single word."}
_JSON_SYSTEM = "Answer with a JSON object."
_JSON_USER = ("Give your strategy as a JSON object whose keys are the legal actions and whose values are "
              "the probabilities you play them with, summing to 1. Reply with only the JSON object.")


def render_all(game: Game, template: str = "A") -> list[Prompt]:
    return [render(game, info, template) for info in game.infosets]


def parse_action_word(text: str, legal: tuple[str, ...]) -> str | None:
    """The legal action named by the first word of a free-text answer, or None.

    Markdown emphasis, quotes and trailing punctuation are ignored, as is case.
    "Call." and "**call**" count as call; "I call" does not.
    """
    stripped = text.strip().lstrip("*_`\"'([ ").strip()
    if not stripped:
        return None
    first = stripped.split()[0].strip("*_`\"'.,;:!?)]").lower()
    return first if first in legal else None
