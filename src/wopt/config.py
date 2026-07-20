"""The typed config, loaded from `config/` with OmegaConf.

One `Config` object is threaded through the whole run, so every knob has exactly
one home and shows up in `config/config.yaml`. The dataclasses below are the
schema: OmegaConf validates the YAML against them, so a typo in a key or a
string where a number belongs fails at load time rather than mid-run.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import OmegaConf

from .paths import CONFIG_DIR


@dataclass
class ModelSpec:
    """One model the optimizer may use: its price and whether it can think."""
    id: str
    price_in: float          # USD per 1,000,000 input tokens
    price_out: float         # USD per 1,000,000 output tokens
    thinks: bool = False     # supports the effort / adaptive-thinking params


@dataclass
class CallConfig:
    max_output_tokens: int = 64000
    max_tool_turns: int = 5
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.10


@dataclass
class RuntimeConfig:
    max_calls: int = 24
    token_budget: int = 120_000
    workers: int = 8
    sandbox: bool = True


@dataclass
class JudgeConfig:
    model: str = "claude-haiku-4-5"
    min_gold: float = 0.7
    max_empty: float = 0.5


@dataclass
class DataConfig:
    n_examples: int = 100
    dev_fraction: float = 0.6
    case_types: int = 15
    batch_size: int = 20
    judge_batch_size: int = 5
    max_stalls: int = 3


@dataclass
class DesignerConfig:
    model: str = "claude-opus-4-8"
    rounds: int = 3
    dev_sample: int = 5
    skills: list[str] = field(
        default_factory=lambda: ["workflow-design", "workflow-eval", "workflow-naming"])
    allowed_tools: list[str] = field(
        default_factory=lambda: ["WebSearch", "WebFetch", "Bash", "Read", "Write", "Skill"])


@dataclass
class TaskConfig:
    """The only per-task input: a description, and optionally your own data."""
    name: str = "custom"
    seed_prompt: str = ""
    dataset: Optional[str] = None    # .jsonl of {"question", "answer"}
    grader: Optional[str] = None     # .py exposing grade(prediction, item) -> float


@dataclass
class ReportConfig:
    budget: float = 0.002
    accuracy_target: float = 0.80
    output_dir: str = "runs"


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    models: list[ModelSpec] = field(default_factory=list)
    analysis_model: str = "claude-opus-4-8"
    call: CallConfig = field(default_factory=CallConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    designer: DesignerConfig = field(default_factory=DesignerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def load_config(task: str = "clinical_notes", overrides: list[str] = ()) -> Any:
    """`config/config.yaml`, then `config/task/<task>.yaml`, then `a.b=c` overrides.

    A task file may set any key, not just `task.*` — a benchmark that needs a
    tighter per-query budget says so next to its own description.
    """
    cfg = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(CONFIG_DIR / "config.yaml"),
    )
    if task:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "task" / f"{task}.yaml"))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
