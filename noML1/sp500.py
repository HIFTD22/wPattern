python - << 'PY'
import pandas as pd
import sys, re

# Pull from Wikipedia (most up to date public source)
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)
symbols = tables[0]["Symbol"].astype(str).str.strip().tolist()

# Normalize for Yahoo Finance (e.g., BRK.B -> BRK-B)
def yf_norm(s): return s.upper().replace('.', '-')

seen = set()
out = []
for s in symbols:
    s = yf_norm(s)
    if s and s not in seen:
        seen.add(s)
        out.append(s)

print("self.sp500_symbols = [")
for i, s in enumerate(out):
    comma = "," if i < len(out)-1 else ""
    print(f"    '{s}'{comma}")
print("]")
print(f"# Total: {len(out)} tickers")
PY
