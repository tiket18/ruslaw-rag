import pyarrow.parquet as pq
import glob
from collections import Counter

base = "/root/.cache/huggingface/hub/datasets--irlspbru--RusLawOD/snapshots/f850b966648499d7ff4f4bc3ef2cddb68f4ec3c0/"
files = sorted(glob.glob(base + "ruslawod_*.parquet"))
files = files[:1]

doc_types = Counter()
issuers = Counter()

for f in files:
    tbl = pq.read_table(f, columns=["doc_typeIPS", "issuedByIPS", "headingIPS", "textIPS", "statusIPS"])
    doc_types.update(tbl.column("doc_typeIPS").to_pylist())
    issuers.update(tbl.column("issuedByIPS").to_pylist())

print("Document types:")
for k,v in doc_types.most_common(30):
    print(f"  {k}: {v}")

print("\nTop issuers:")
for k,v in issuers.most_common(30):
    print(f"  {k}: {v}")

print("\n=== Постановления Пленума ВС / Верховный Суд ===")
for i in range(tbl.num_rows):
    issuer = str(tbl.column("issuedByIPS")[i].as_py() or "")
    heading = str(tbl.column("headingIPS")[i].as_py() or "")
    text = str(tbl.column("textIPS")[i].as_py() or "")
    h = heading.lower()
    if any(w in issuer.lower() for w in ["верховн", "пленум", "конституцион"]):
        print(f"  [{tbl.column('statusIPS')[i].as_py()}] {issuer} | {heading[:120]} | text_len={len(text)}")

print("\n=== Кодексы (полный текст, не поправки) ===")
for i in range(tbl.num_rows):
    issuer = str(tbl.column("issuedByIPS")[i].as_py() or "")
    heading = str(tbl.column("headingIPS")[i].as_py() or "")
    text = str(tbl.column("textIPS")[i].as_py() or "")
    h = heading.lower()
    doc_type = str(tbl.column("doc_typeIPS")[i].as_py() or "")
    status = str(tbl.column("statusIPS")[i].as_py() or "")
    # Look for actual codes (not just amendments mentioning codes)
    codes = ["гражданский кодекс", "налоговый кодекс", "жилищный кодекс", "трудовой кодекс",
             "гражданский процессуальный", "арбитражный процессуальный", "кодекс об административных",
             "уголовный кодекс", "уголовно-процессуальный", "бюджетный кодекс", "водный кодекс",
             "лесной кодекс", "земельный кодекс", "семейный кодекс", "воздушный кодекс"]
    if doc_type == "Федеральный закон" and "кодекс" in h and len(text) > 50000:
        # This might be a code itself, not an amendment
        is_amendment = any(w in h for w in ["внесен", "изменен", "признан"])
        print(f"  [{'amend' if is_amendment else 'CODE'}] [{status}] {heading[:150]} | text_len={len(text)}")
