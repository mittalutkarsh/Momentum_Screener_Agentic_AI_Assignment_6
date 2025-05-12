from pydantic import BaseModel
from typing import List, Optional
import code

# ← Define this before you reference it
class Stock(BaseModel):
    symbol: str
    price: float
    volume: int
    # add any Optional[...] fields here, e.g.:
    # high_52_week: Optional[float] = None

class PerceptionOutput(BaseModel):
    reasoning: str
    analysis: str
    suggestions: str
    result: List[Stock]  # now Stock is known

if __name__ == "__main__":
    output = PerceptionOutput(
        reasoning="The stock price is too high",
        analysis="The stock is overvalued",
        suggestions="Buy the stock",
        result=[Stock(symbol="AAPL", price=150, volume=1_000_000)]
    )
    code.interact(local=locals()) # for debugging

    print(output.model_dump_json())

