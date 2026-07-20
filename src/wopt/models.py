"""The model catalog: which models exist, what they cost, what they can do."""
from dataclasses import dataclass


@dataclass
class Catalog:
    """Everything model-related, derived from `config.models`.

    Cost and capability are looked up in the same list the search routes over,
    so a price and the model it prices cannot drift apart.
    """
    specs: list           # ModelSpec, cheapest -> most expensive
    cache_write_multiplier: float
    cache_read_multiplier: float

    @classmethod
    def from_config(cls, cfg) -> "Catalog":
        return cls(specs=list(cfg.models),
                   cache_write_multiplier=cfg.call.cache_write_multiplier,
                   cache_read_multiplier=cfg.call.cache_read_multiplier)

    @property
    def ids(self) -> list[str]:
        """Model ids, cheapest -> most expensive: the search pool, and the menu a
        workflow routes over."""
        return [m.id for m in self.specs]

    @property
    def default(self) -> str:
        """A workflow's starting model — the cheapest. It may escalate itself."""
        return self.ids[0]

    def spec(self, model_id: str):
        for m in self.specs:
            if m.id == model_id:
                return m
        return None

    def thinks(self, model_id: str) -> bool:
        spec = self.spec(model_id)
        return bool(spec and spec.thinks)

    def resolve(self, model_id: str | None) -> str:
        """An unknown or missing model falls back to the default rather than
        erroring: model-written code routes by name and may invent one."""
        return model_id if self.spec(model_id) else self.default

    def cost_usd(self, model_id: str, usage: dict) -> float:
        """Price one call's token usage. Cache-aware: a write costs a little more
        than fresh input, a read ~90% less."""
        spec = self.spec(model_id)
        tokens = (usage["input"] * spec.price_in
                  + usage["cache_write"] * spec.price_in * self.cache_write_multiplier
                  + usage["cache_read"] * spec.price_in * self.cache_read_multiplier
                  + usage["output"] * spec.price_out)
        return tokens / 1_000_000
