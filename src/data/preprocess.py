import re
import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Reddit comments/posts:
    - Lowercase text
    - Remove URLs + special chars
    - Collapse multiple spaces
    - Apply regex-based labeling
    """

    if df.empty:
        return pd.DataFrame(columns=["text", "label"])

    # --- Clean text ---
    df["text"] = df["text"].astype(str).str.lower()
    df["text"] = df["text"].str.replace(r"http\S+", " ", regex=True)     # remove URLs
    df["text"] = df["text"].str.replace(r"[^a-z0-9\s]", " ", regex=True) # keep alphanum + spaces
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # --- Label patterns (expanded) ---
    # Student Visa Stage
    student_pat = r"(\bf1\b|\bcpt\b|\bopt\b|stem\s*opt|visa\s*(interview|stamp|stamping|appointment)|i-765|work\s*authoriz|visa\s*renew|visa\s*status)"
    # Work Visa Stage
    work_pat    = r"(\bh-?1b\b|\bh1b\b|employ(er|ee)\s*(sponsor|withdraw|withdrawal)|job\s*search\s*visa|visa\s*denial|visa\s*delay|immigration\s*backlog|policy\s*change)"
    # Permanent Residency Stage
    perm_pat    = r"(i-140|perm|green\s*card|\bgc\b|adjustment\s*of\s*status|consular\s*processing|priority\s*date|visa\s*bulletin|\brfe\b|i-485|uscis\s*case\s*status)"
    # Citizenship Stage
    cit_pat     = r"(citizenship|naturalization|immigration\s*reform|travel\s*ban|deportation\s*risk|policy\s*change)"
    # General Immigration & Legal Issues (+ political terms)
    general_pat = r"(visa\s*denial|visa\s*delays?|immigration\s*backlog|immigration\s*reform|legal\s*help|work\s*authorization|immigration\s*policy|delay|issue|deny|denied|denies|out\s*of\s*time|trump|administration|executive|\beo\b)"

    subreddit_to_label = {
        # Student
        "f1visa": "student_visa",
        "opt": "student_visa",
        "stemopt": "student_visa",
        # Work
        "h1b": "work_visa",
        "workvisas": "work_visa",
        # Permanent Residency
        "greencard": "green_card",
        "greencardprocess": "green_card",
        # Citizenship
        "citizenship": "citizenship",
    }

    def assign_label(text: str, subreddit: str) -> str:
        sub = (subreddit or "").lower()
        if sub in subreddit_to_label:
            return subreddit_to_label[sub]
        if re.search(student_pat, text): return "student_visa"
        if re.search(work_pat, text): return "work_visa"
        if re.search(perm_pat, text): return "green_card"
        if re.search(cit_pat, text): return "citizenship"
        if re.search(general_pat, text): return "general_immigration"
        return "irrelevant"

    subreddit_col = None
    for c in ["subreddit", "subreddit_name_prefixed", "subreddit_id"]:
        if c in df.columns:
            subreddit_col = c
            break
    if subreddit_col is None:
        df["label"] = df["text"].apply(lambda t: assign_label(t, ""))
    else:
        df["label"] = [assign_label(t, str(s)) for t, s in zip(df["text"], df[subreddit_col])]

    # Drop rows with empty text
    df = df[df["text"].str.len() > 3]

    return df[["text", "label"]]
