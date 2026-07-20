"""Analyze full RusLawOD dataset: codes, supreme court, government."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
import glob
from collections import Counter

BASE = "/home/vllm/rag/raw/"
files = sorted(glob.glob(BASE + "ruslawod_*.parquet"))
print(f"Files: {len(files)}")

total = 0
total_text_size = 0
codes = []
supreme_court = []
gov_regs = []
active_gov = 0
by_type = Counter()
by_issuer = Counter()

for f in files:
    tbl = pq.read_table(f, columns=[
        "doc_typeIPS", "issuedByIPS", "headingIPS", "docNumberIPS",
        "docdateIPS", "statusIPS", "textIPS", "actual_datetimeIPS"
    ])
    total += tbl.num_rows
    
    for i in range(tbl.num_rows):
        dtype = str(tbl.column("doc_typeIPS")[i].as_py() or "")
        issuer = str(tbl.column("issuedByIPS")[i].as_py() or "")
        heading = str(tbl.column("headingIPS")[i].as_py() or "")
        status = str(tbl.column("statusIPS")[i].as_py() or "")
        text = str(tbl.column("textIPS")[i].as_py() or "")
        docnum = str(tbl.column("docNumberIPS")[i].as_py() or "")
        docdate = str(tbl.column("docdateIPS")[i].as_py() or "")
        h = heading.lower()
        total_text_size += len(text)
        
        by_type[dtype] += 1
        by_issuer[issuer] += 1
        
        # Find actual codes (not just amendments)
        is_code = dtype == "Кодекс" or (
            dtype == "Федеральный закон" and "кодекс" in h and "внесен" not in h
        )
        if is_code and len(text) > 5000:
            codes.append((issuer, heading, docnum, docdate, status, len(text)))
        
        # Supreme Court and Constitutional Court practice
        if any(w in issuer for w in ["Верховн", "Конституцион", "Пленум"]):
            supreme_court.append((issuer, heading, status, len(text)))
        
        # Government regulations
        if "Правительство" in issuer and dtype == "Постановление":
            gov_regs.append((issuer, heading, status, len(text), docdate))
            if status and "Действует" in status:
                active_gov += 1

print(f"\nTotal documents: {total}")
print(f"Total text volume: ~{total_text_size // 1024 // 1024} MB")

print(f"\n=== КОДЕКСЫ (полные тексты) ===")
print(f"Найдено: {len(codes)}")
for iss, head, num, date, status, tlen in sorted(codes, key=lambda x: -x[5])[:30]:
    print(f"  [{status}] {iss} | {head[:120]} | {num} от {date} | {tlen//1024} KB")

print(f"\n=== ВЕРХОВНЫЙ СУД / КОНСТИТУЦИОННЫЙ СУД ===")
print(f"Найдено: {len(supreme_court)}")
for iss, head, status, tlen in sorted(supreme_court, key=lambda x: -x[3])[:30]:
    print(f"  [{status}] {iss} | {head[:120]} | {tlen//1024} KB")

print(f"\n=== ПОСТАНОВЛЕНИЯ ПРАВИТЕЛЬСТВА ===")
print(f"Всего: {len(gov_regs)}, из них действует: {active_gov}")
by_year = Counter()
for iss, head, status, tlen, date in gov_regs:
    if date and len(date) >= 4:
        by_year[date[:4]] += 1
print("По годам:")
for y in sorted(by_year):
    print(f"  {y}: {by_year[y]}")

print(f"\n=== ТОП-20 ТИПОВ ДОКУМЕНТОВ ===")
for k,v in by_type.most_common(20):
    print(f"  {k}: {v}")

print(f"\n=== ТОП-20 ИЗДАТЕЛЕЙ ===")
for k,v in by_issuer.most_common(20):
    print(f"  {k}: {v}")
