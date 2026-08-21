import os

# $ per million tokens (input, output). Update as pricing changes.
PRICING_PER_MTOK = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-fable-5": (0.25, 1.25),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}
DEFAULT_PRICING = (3.00, 15.00)

DEFAULT_MODEL = os.environ.get("AGENTBENCH_MODEL", "claude-sonnet-5")
DEFAULT_MAX_ITERS = int(os.environ.get("AGENTBENCH_MAX_ITERS", "3"))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICING_PER_MTOK.get(model, DEFAULT_PRICING)
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
