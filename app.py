import os
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime

from flask import Flask, request, jsonify, Response, send_file

import pandas as pd
import numpy as np

# ---------------------------------
# Flask setup
# ---------------------------------
# Serve index.html and styles.css from current directory
app = Flask(__name__, static_url_path="", static_folder=".")

# Global in-memory state
DF = None            # enriched DataFrame
LAST_PICKS = None    # list of dicts for most recent 5 selections

# ---------------------------------
# Config and heuristics
# ---------------------------------
ROLE_KEYWORDS = {
    "engineering": {
        "python","java","go","golang","rust","c++","c#","typescript","javascript","node","react","next",
        "vue","svelte","django","flask","fastapi","rails","spring","kotlin","swift",
        "aws","gcp","azure","docker","kubernetes","terraform","ansible","postgres","mysql","redis",
        "graphql","rest","grpc","kafka","spark","hadoop","airflow","ci","cd","linux",
        "ml","ai","deep","pytorch","tensorflow","nlp","cv","rag","llm","langchain","vectordb","embedding"
    },
    "data": {
        "sql","python","r","pandas","numpy","scipy","dbt","airflow","dagster","spark","hive","presto",
        "bigquery","redshift","snowflake","databricks","etl","elt","warehouse","lakehouse","tableau","looker","powerbi",
        "cohort","experiment","ab","metrics","analytics","forecast","modeling"
    },
    "product": {
        "product","pm","product manager","roadmap","spec","prd","discovery","user research",
        "kpi","experimentation","growth","requirements","prioritization","stakeholder","scrum","agile"
    },
    "design": {
        "design","ux","ui","product design","interaction","figma","sketch","framer","prototyping","usability",
        "visual","motion","illustration","design system","accessibility","a11y"
    },
    "gtm": {
        "sales","business development","bd","partnerships","marketing","growth","seo","sem","ads","crm",
        "hubspot","salesforce","pipeline","quota","lead","content","copy","demand","kpi"
    },
    "ops": {
        "ops","operations","finance","fp&a","legal","people","hr","recruiting","talent","compliance","support","success"
    },
}

SENIORITY_WORDS = {
    "intern": 0, "junior": 1, "jr": 1, "associate": 2, "mid": 2, "middle": 2,
    "senior": 3, "sr": 3, "lead": 4, "staff": 4, "principal": 5,
    "manager": 4, "head": 5, "director": 5, "vp": 6,
    "founder": 6, "cto": 6, "cpo": 6, "coo": 6
}

IMPACT_WORDS = [
    "shipped","launched","scale","scaled","optimize","optimized","grew","increase","decrease","reduced",
    "revenue","arr","mrr","retention","conversion","throughput","latency","cost","nps","engagement",
    "dau","mau","adoption","acquired","sold","raised","funded","patent","open source","oss","publication","paper"
]

TOP_COMPANIES = {
    "google","alphabet","meta","facebook","amazon","apple","netflix","microsoft","openai","stripe",
    "airbnb","uber","lyft","dropbox","salesforce","snowflake","nvidia","databricks","coinbase","palantir"
}

TOP_SCHOOLS = {
    "stanford","mit","massachusetts institute of technology","berkeley","uc berkeley","university of california, berkeley",
    "harvard","caltech","cmu","carnegie mellon","waterloo","oxford","cambridge","eth zurich","ucla","uw","illinois","uiuc"
}

REGION_KEYWORDS = {
    "north america": ["usa","united states","canada","mexico","ca","us","ny","sf","san francisco","seattle","austin","boston","nyc","toronto","vancouver","montreal"],
    "europe": ["uk","united kingdom","england","germany","france","spain","italy","netherlands","sweden","norway","denmark","finland","poland","ireland","portugal","estonia","lithuania","latvia","switzerland"],
    "latam": ["brazil","argentina","chile","peru","colombia","uruguay","paraguay","ecuador","bolivia","venezuela"],
    "asia": ["india","singapore","malaysia","indonesia","philippines","vietnam","thailand","japan","korea","south korea","taiwan","china","hong kong"],
    "africa": ["nigeria","kenya","ghana","egypt","south africa","ethiopia","morocco","tunisia","algeria"],
    "oceania": ["australia","new zealand","sydney","melbourne","auckland","brisbane"]
}

DEFAULT_BUCKET_PLAN = [
    ("engineering", 2),
    ("product", 1),
    ("design", 1),
    ("gtm", 1),  # falls back to data if no GTM candidates exist
]

# ---------------------------------
# Helpers
# ---------------------------------
def norm_text(x):
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        try:
            x = json.dumps(x)
        except Exception:
            x = str(x)
    x = unicodedata.normalize("NFKC", str(x))
    return x.strip()

def token_set(text):
    t = norm_text(text).lower()
    toks = re.findall(r"[a-zA-Z0-9\+#\.]+", t)
    return set(toks)

def find_first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return None

def extract_years(text):
    t = norm_text(text).lower()
    m = re.findall(r"(\d+)\+?\s*years?", t)
    vals = [int(v) for v in m if v.isdigit() and int(v) < 60]
    if vals:
        return max(vals)
    return None

def has_any(text, words):
    t = norm_text(text).lower()
    return any(w.lower() in t for w in words)

def count_hits(text, words):
    t = norm_text(text).lower()
    return sum(t.count(w.lower()) for w in words)

def guess_region(location):
    t = norm_text(location).lower()
    for region, kws in REGION_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return region
    return "unknown"

def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

def load_json(file_bytes: bytes):
    raw = file_bytes.decode("utf-8", errors="ignore")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):  # single object
            data = [data]
    except Exception:
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data

def normalize_rows(rows):
    out = []
    for i, r in enumerate(rows):
        # Name handling (first + last if needed)
        name = find_first(r, ["name","full_name","fullName","candidateName"])
        if not name:
            first = find_first(r, ["first_name","firstName"])
            last = find_first(r, ["last_name","lastName","surname"])
            if first or last:
                name = f"{norm_text(first)} {norm_text(last)}".strip()

        email = find_first(r, ["email","email_address","contact_email"])
        location = find_first(r, ["location","city","country","timezone","time_zone"])
        linkedin = find_first(r, ["linkedin","linkedin_url","LinkedIn"])
        github = find_first(r, ["github","github_url"])
        portfolio = find_first(r, ["portfolio","website","site","portfolio_url","personal_website"])
        role_pref = find_first(r, ["role","desired_role","role_preference","interested_in","position","title"])
        skills = find_first(r, ["skills","skillset","tech_stack","technologies","top_skills","stack"])
        summary = (
            find_first(r, ["about","bio","summary","notes","pitch","headline","profileSummary","short_intro","cover_letter"]) or
            find_first(r, ["resume_text","experience_summary"])
        )
        years = find_first(r, ["years_experience","yoe","total_experience","experience_years"])
        companies = find_first(r, ["companies","work_history","experience","current_company","previous_companies","employment_history"])
        education = find_first(r, ["education","school","degree","university"])
        resume = find_first(r, ["resume","resume_url","cv","cv_url"])

        # Extract years if text
        y_numeric = None
        if isinstance(years, (int, float)):
            y_numeric = float(years)
        else:
            y_numeric = extract_years(years) or extract_years(summary) or None

        # Skills text normalize
        if isinstance(skills, list):
            skill_text = ", ".join([norm_text(s) for s in skills])
        else:
            skill_text = norm_text(skills)

        merged_text = " | ".join(filter(None, [
            norm_text(role_pref), skill_text, norm_text(summary), norm_text(companies), norm_text(education)
        ]))

        out.append({
            "id": i,
            "name": norm_text(name) or f"Candidate #{i+1}",
            "email": norm_text(email),
            "location": norm_text(location),
            "linkedin": norm_text(linkedin),
            "github": norm_text(github),
            "portfolio": norm_text(portfolio),
            "role_preference": norm_text(role_pref),
            "skills_raw": skill_text,
            "summary": norm_text(summary),
            "years_experience": y_numeric if y_numeric is not None else np.nan,
            "companies_raw": norm_text(companies),
            "education_raw": norm_text(education),
            "resume_link": norm_text(resume),
            "merged_text": merged_text
        })
    return pd.DataFrame(out)

def guess_roles_and_scores(row):
    skill_tokens = token_set(row["skills_raw"]) | token_set(row["role_preference"]) | token_set(row["summary"])
    role_scores = {}
    for role, kws in ROLE_KEYWORDS.items():
        role_scores[role] = jaccard(skill_tokens, set(kws))
    best_role = max(role_scores, key=role_scores.get) if role_scores else "engineering"
    return best_role, role_scores

def score_candidate(row, role_scores):
    best_role_score = max(role_scores.values()) if role_scores else 0.0
    y = row.get("years_experience")
    if pd.isna(y):
        exp_score = 0.25
    else:
        exp_score = min(y / 10.0, 1.0)
        if 3 <= y <= 10:
            exp_score = min(exp_score + 0.15, 1.0)
    impact_hits = count_hits(row["merged_text"], IMPACT_WORDS)
    impact_score = min(impact_hits / 5.0, 1.0)
    github_score = 1.0 if row["github"] else 0.0
    portfolio_score = 1.0 if row["portfolio"] else 0.0
    resume_score = 1.0 if row["resume_link"] else 0.0
    leadership_score = 1.0 if has_any(row["merged_text"], SENIORITY_WORDS.keys()) else 0.0
    ped_company = 1.0 if has_any(row["companies_raw"], TOP_COMPANIES) else 0.0
    ped_school = 1.0 if has_any(row["education_raw"], TOP_SCHOOLS) else 0.0
    pedigree_score = max(ped_company, ped_school)
    edu_score = 1.0 if has_any(row["education_raw"], ["phd","ms","msc","masters","m.s.","m.s","m.eng","mba"]) else 0.0
    summary_len = len(row["summary"])
    writing_score = max(0.0, min(summary_len / 1000.0, 1.0))
    final = (
        0.40 * best_role_score +
        0.15 * exp_score +
        0.12 * impact_score +
        0.08 * github_score +
        0.05 * portfolio_score +
        0.03 * resume_score +
        0.05 * leadership_score +
        0.05 * pedigree_score +
        0.02 * edu_score +
        0.05 * writing_score
    )
    return float(round(final, 4))

def enrich(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    best_roles, role_distrib, region_list, finals = [], [], [], []
    for _, row in df.iterrows():
        best_role, role_scores = guess_roles_and_scores(row)
        best_roles.append(best_role)
        role_distrib.append({k: float(v) for k, v in role_scores.items()})
        region_list.append(guess_region(row["location"]))
        finals.append(score_candidate(row, role_scores))
    df = df.copy()
    df["role_guess"] = best_roles
    df["role_scores"] = role_distrib
    df["region"] = region_list
    df["final_score"] = finals
    return df

def pick_diverse_five(df: pd.DataFrame, prefer_regions=True, plan=None):
    if plan is None:
        plan = DEFAULT_BUCKET_PLAN.copy()
    if df is None or df.empty:
        return []

    have_roles = set(df["role_guess"].unique().tolist())
    plan2 = []
    for role, n in plan:
        if role == "gtm" and "gtm" not in have_roles and "data" in have_roles:
            plan2.append(("data", n))
        else:
            plan2.append((role, n))

    picks = []
    taken_ids = set()

    for role, slots in plan2:
        role_pool = df[df["role_guess"] == role].sort_values("final_score", ascending=False)
        for _, row in role_pool.iterrows():
            if len([p for p in picks if p["role_guess"] == role]) >= slots:
                break
            if row["id"] in taken_ids:
                continue
            picks.append(row.to_dict())
            taken_ids.add(row["id"])
            if len(picks) >= 5:
                break
        if len(picks) >= 5:
            break

    if len(picks) < 5:
        remaining = df[~df["id"].isin(taken_ids)].sort_values("final_score", ascending=False)
        for _, row in remaining.iterrows():
            picks.append(row.to_dict())
            taken_ids.add(row["id"])
            if len(picks) >= 5:
                break

    if prefer_regions and picks:
        region_counts = Counter([p["region"] for p in picks])
        if len(region_counts) == 1:
            last = picks[-1]
            alt = df[(~df["id"].isin(taken_ids)) & (df["region"] != last["region"])].sort_values("final_score", ascending=False)
            if not alt.empty:
                picks[-1] = alt.iloc[0].to_dict()

    return picks[:5]

def reasons_for_candidate(c):
    rs = []
    role = c.get("role_guess", "—")
    y = c.get("years_experience", np.nan)
    if not pd.isna(y):
        rs.append(f"{int(y)} years of experience; strong for early-stage.")
    role_score = c.get("role_scores", {}).get(role, 0.0)
    if role_score >= 0.3:
        rs.append(f"High skill-role match for {role} (fit score {role_score:.2f}).")
    if c.get("github"):
        rs.append("Public GitHub profile indicates shipping/OSS exposure.")
    if c.get("portfolio"):
        rs.append("Portfolio/website shows artifacts and taste.")
    if has_any(c.get("companies_raw",""), TOP_COMPANIES):
        rs.append("Experience at high-signal company (pedigree).")
    if has_any(c.get("merged_text",""), ["lead","manager","founder","head","director","staff","principal"]):
        rs.append("Leadership experience; can own large scope.")
    if has_any(c.get("education_raw",""), ["phd","master","msc","mba"]):
        rs.append("Advanced degree (bonus).")
    impacts = count_hits(c.get("merged_text",""), IMPACT_WORDS)
    if impacts >= 2:
        rs.append("Multiple impact indicators (shipped/launched/scaled).")
    if c.get("region") and c.get("region") != "unknown":
        rs.append(f"Adds geographic/timezone diversity ({c['region']}).")
    if not rs:
        rs = ["Strong composite score based on skills, experience, and signals."]
    return rs[:6]

def to_native(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def df_to_records(df, cols=None):
    if df is None or df.empty:
        return []
    if cols is None:
        cols = df.columns.tolist()
    recs = []
    for _, r in df[cols].iterrows():
        item = {}
        for c in cols:
            v = r[c]
            if isinstance(v, dict):
                item[c] = {k: to_native(val) for k, val in v.items()}
            else:
                item[c] = to_native(v)
        recs.append(item)
    return recs

# ---------------------------------
# Routes
# ---------------------------------
@app.route("/")
def index():
    # Serves index.html from current folder
    return app.send_static_file("index.html")

@app.route("/api/config", methods=["GET"])
def config():
    roles = list(ROLE_KEYWORDS.keys())
    plan = DEFAULT_BUCKET_PLAN
    return jsonify({"roles": roles, "default_plan": plan})

@app.route("/api/upload", methods=["POST"])
def upload():
    global DF, LAST_PICKS
    LAST_PICKS = None
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (key 'file')"}), 400
    f = request.files["file"]
    if not f:
        return jsonify({"error": "Empty file"}), 400

    try:
        rows = load_json(f.read())
        if not rows:
            return jsonify({"error": "Could not parse any JSON objects."}), 400
        df = normalize_rows(rows)
        df = enrich(df)
        DF = df
        # Summary
        roles = df["role_guess"].value_counts().to_dict()
        regions = df["region"].value_counts().to_dict()
        summary = {
            "total": int(len(df)),
            "avg_score": float(round(df["final_score"].mean(), 4)) if len(df) else 0.0,
            "roles": roles,
            "regions": regions,
        }
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {e}"}), 500

@app.route("/api/summary", methods=["GET"])
def summary():
    if DF is None or DF.empty:
        return jsonify({"error": "No dataset loaded yet."}), 400
    df = DF
    roles = df["role_guess"].value_counts().to_dict()
    regions = df["region"].value_counts().to_dict()
    top_role = max(roles, key=roles.get) if roles else None
    top_region = max(regions, key=regions.get) if regions else None
    return jsonify({
        "total": int(len(df)),
        "avg_score": float(round(df["final_score"].mean(), 4)) if len(df) else 0.0,
        "roles": roles,
        "regions": regions,
        "top_role": top_role,
        "top_region": top_region
    })

@app.route("/api/candidates", methods=["GET"])
def candidates():
    if DF is None or DF.empty:
        return jsonify({"error": "No dataset loaded yet."}), 400

    df = DF.copy()

    # basic filters
    try:
        min_score = float(request.args.get("min_score", "0") or "0")
    except:
        min_score = 0.0
    roles_param = request.args.get("roles", "")
    roles_filter = [r.strip().lower() for r in roles_param.split(",") if r.strip()] or list(ROLE_KEYWORDS.keys())

    # advanced filters (optional)
    search = (request.args.get("search") or "").strip().lower()
    regions_param = request.args.get("regions", "")
    regions_filter = [r.strip().lower() for r in regions_param.split(",") if r.strip()]

    yoe_min = request.args.get("yoe_min")
    yoe_max = request.args.get("yoe_max")
    yoe_min = None if yoe_min in (None, "",) else float(yoe_min)
    yoe_max = None if yoe_max in (None, "",) else float(yoe_max)

    has_github = request.args.get("has_github", "false").lower() == "true"
    has_portfolio = request.args.get("has_portfolio", "false").lower() == "true"
    has_resume = request.args.get("has_resume", "false").lower() == "true"
    ped_company = request.args.get("ped_company", "false").lower() == "true"
    ped_school = request.args.get("ped_school", "false").lower() == "true"

    sort_by = (request.args.get("sort_by") or "final_score").strip()
    sort_dir = (request.args.get("sort_dir") or "desc").strip().lower()
    ascending = sort_dir == "asc"

    # apply filters
    df = df[(df["final_score"] >= min_score) & (df["role_guess"].isin(roles_filter))]

    if regions_filter:
      df = df[df["region"].str.lower().isin(regions_filter)]

    if yoe_min is not None:
      df = df[df["years_experience"].fillna(-1) >= yoe_min]
    if yoe_max is not None:
      df = df[df["years_experience"].fillna(1e9) <= yoe_max]

    if has_github:
      df = df[df["github"].fillna("") != ""]
    if has_portfolio:
      df = df[df["portfolio"].fillna("") != ""]
    if has_resume:
      df = df[df["resume_link"].fillna("") != ""]

    if ped_company:
      mask = df["companies_raw"].fillna("").str.contains("|".join(sorted(TOP_COMPANIES, key=len, reverse=True)), case=False, regex=True)
      df = df[mask]
    if ped_school:
      mask = df["education_raw"].fillna("").str.contains("|".join(sorted(TOP_SCHOOLS, key=len, reverse=True)), case=False, regex=True)
      df = df[mask]

    if search:
      hay = (df["merged_text"].fillna("") + " " + df["skills_raw"].fillna("") + " " + df["location"].fillna(""))
      df = df[hay.str.contains(re.escape(search), case=False, regex=True)]

    # sorting
    if sort_by not in df.columns:
        sort_by = "final_score"
    df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")  # stable

    cols = [
        "id","name","role_guess","final_score","years_experience","region","location",
        "email","linkedin","github","portfolio","resume_link","skills_raw","companies_raw","education_raw"
    ]
    recs = df_to_records(df, cols)

    # Summary for filtered view
    role_counts = df["role_guess"].value_counts().to_dict()
    region_counts = df["region"].value_counts().to_dict()
    return jsonify({
        "candidates": recs,
        "filtered": {
            "count": int(len(df)),
            "avg_score": float(round(df["final_score"].mean(), 4)) if len(df) else 0.0,
            "roles": role_counts,
            "regions": region_counts
        }
    })
@app.route("/api/candidate", methods=["GET"])
def candidate_detail():
    if DF is None or DF.empty:
        return jsonify({"error": "No dataset loaded yet."}), 400
    try:
        cid = int(request.args.get("id"))
    except:
        return jsonify({"error": "Missing or invalid id"}), 400

    row = DF[DF["id"] == cid]
    if row.empty:
        return jsonify({"error": "Candidate not found"}), 404
    r = row.iloc[0].to_dict()
    # Prepare detail
    detail = {
        "id": int(r["id"]),
        "name": r["name"],
        "email": r.get("email"),
        "location": r.get("location"),
        "region": r.get("region"),
        "role_guess": r.get("role_guess"),
        "years_experience": None if pd.isna(r.get("years_experience")) else int(r.get("years_experience")),
        "final_score": float(r.get("final_score")),
        "links": {
            "linkedin": r.get("linkedin"),
            "github": r.get("github"),
            "portfolio": r.get("portfolio"),
            "resume": r.get("resume_link"),
        },
        "skills_raw": r.get("skills_raw"),
        "summary": r.get("summary"),
        "companies_raw": r.get("companies_raw"),
        "education_raw": r.get("education_raw"),
        "role_scores": {k: float(v) for k, v in r.get("role_scores", {}).items()},
        "merged_text": r.get("merged_text"),
        "reasons": reasons_for_candidate(r)
    }
    return jsonify(detail)

@app.route("/api/auto_select", methods=["POST"])
def auto_select():
    global LAST_PICKS
    if DF is None or DF.empty:
        return jsonify({"error": "No dataset loaded yet."}), 400

    data = request.json or {}
    try:
        min_score = float(data.get("min_score", 0.0))
    except:
        min_score = 0.0
    roles = data.get("roles") or list(ROLE_KEYWORDS.keys())
    roles = [r.strip().lower() for r in roles if r.strip()]
    prefer_regions = bool(data.get("prefer_regions", True))

    df = DF[(DF["final_score"] >= min_score) & (DF["role_guess"].isin(roles))].copy()
    df = df.sort_values("final_score", ascending=False)
    picks = pick_diverse_five(df, prefer_regions=prefer_regions)

    # attach reasons
    for p in picks:
        p["reasons"] = reasons_for_candidate(p)
        # sanitize types for JSON
        p["final_score"] = float(p.get("final_score", 0.0))
        if not pd.isna(p.get("years_experience", np.nan)):
            p["years_experience"] = int(p["years_experience"])
        else:
            p["years_experience"] = None
        p["role_scores"] = {k: float(v) for k, v in p.get("role_scores", {}).items()}

    LAST_PICKS = picks
    return jsonify({"picks": picks})

@app.route("/api/export", methods=["GET"])
def export_csv():
    if not LAST_PICKS:
        return jsonify({"error": "No picks to export. Run Auto-select first."}), 400
    cols = ["name","role_guess","final_score","years_experience","region","location","email","linkedin","github","portfolio","resume_link","skills_raw","summary","companies_raw","education_raw"]
    rows = []
    for p in LAST_PICKS:
        row = {c: p.get(c, "") for c in cols}
        row["final_score"] = p.get("final_score", 0.0)
        row["years_experience"] = p.get("years_experience") or ""
        row["reasons"] = " | ".join(p.get("reasons", []))
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"selected_5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    # host=0.0.0.0 if you want to hit from other devices on LAN
    app.run(host="127.0.0.1", port=5000, debug=True)