"""Parse the raw Upwork RSS export into a clean, readable gigs table."""
import re
import sys

import pandas as pd

raw_path, out_full, out_preview = sys.argv[1:4]
d = pd.read_csv(raw_path)

MARKERS = r"(?:Hourly Range:|Budget:|Posted On:)"


def job_text(desc):
    return re.split(MARKERS, desc, maxsplit=1)[0].replace("\xa0", " ").strip()


def category(desc):
    m = re.search(r"Category:\s*(.*?)(?:Skills:|Location Requirement:|Country:|click to apply)", desc)
    return m.group(1).strip() if m else ""


def skills(desc):
    m = re.search(r"Skills:(.*?)(?:Skills:|Location Requirement:|Country:|click to apply)", desc, re.S)
    if not m:
        return ""
    tags = [t.strip() for t in m.group(1).split(",")]
    return "; ".join(t for t in tags if t)


out = pd.DataFrame({
    "gig_id": d["link"].str.extract(r"(?:~|%7E)(\w+)")[0],
    "title": d["title"].str.strip(),
    "category": d["description"].map(category),
    "skills": d["description"].map(skills),
    "pay_type": d["is_hourly"].map({True: "hourly", False: "fixed", "True": "hourly", "False": "fixed"}).fillna("unknown"),
    "hourly_low_usd": d["hourly_low"],
    "hourly_high_usd": d["hourly_high"],
    "fixed_budget_usd": d["budget"],
    "country": d["country"].fillna(""),
    "published_date": d["published_date"].str[:10],
    "description": d["description"].map(job_text),
    "link": d["link"].str.replace("?source=rss", "", regex=False),
})
out = out.drop_duplicates("gig_id")
out.to_csv(out_full, index=False, encoding="utf-8-sig")
prev = out.head(100).copy()
prev["description"] = prev["description"].str.slice(0, 300)
prev.to_csv(out_preview, index=False, encoding="utf-8-sig")
print(out.shape, (out.skills == "").mean(), (out.category == "").mean(), out.skills.str.count(";").add(1).median())
