import re
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Reddit comments/posts for immigration journey classification.
    
    Args:
        df: DataFrame with text data
        
    Returns:
        DataFrame with cleaned text and labels
    """
    if df.empty:
        return pd.DataFrame(columns=["text", "label"])

    # Clean text
    df["text"] = df["text"].astype(str).str.lower()
    df["text"] = df["text"].str.replace(r"http\S+", " ", regex=True)  # Remove URLs
    df["text"] = df["text"].str.replace(r"[^a-z0-9\s]", " ", regex=True)  # Keep alphanum + spaces
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # --- Label patterns based on immigration journey stages ---
    
    # 1. Student Visa Stage (F1, CPT, OPT, STEM OPT)
    student_pat = r"(\bf1\b|\bcpt\b|\bopt\b|stem\s*opt|visa\s*(interview|stamp|stamping|appointment)|i-765|work\s*authoriz|visa\s*renew|visa\s*status|student\s*visa|f1\s*visa|opt\s*extension|stem\s*opt\s*extension|visa\s*(interview|stamp|authoriz|renew|status|appoint))"
    
    # 2. Work Visa Stage (H1B, Employer Sponsorship)  
    work_pat = r"(\bh-?1b\b|\bh1b\b|employ(er|ee)\s*(sponsor|withdraw|withdrawal)|job\s*search\s*visa|visa\s*denial|visa\s*delay|immigration\s*backlog|policy\s*change|h1b\s*visa|work\s*visa|employment\s*visa|visa\s*(employ|withdraw|denial|delay))"
    
    # 3. Permanent Residency Stage (PERM, I-140, Green Card/GC)
    perm_pat = r"(i-140|perm|green\s*card|\bgc\b|adjustment\s*of\s*status|consular\s*processing|priority\s*date|visa\s*bulletin|\brfe\b|i-485|uscis\s*case\s*status|permanent\s*residency|eb1|eb2|eb3|employment\s*based|n-400)"
    
    # 4. Citizenship Stage
    cit_pat = r"(citizenship|naturalization|immigration\s*reform|travel\s*ban|deportation\s*risk|policy\s*change|naturalization\s*process|citizenship\s*test|n400)"
    
    # 5. General Immigration & Legal Issues (Cross-cutting all stages)
    general_pat = r"(visa\s*denial|visa\s*delays?|immigration\s*backlog|immigration\s*reform|legal\s*help|work\s*authorization|immigration\s*policy|delay|issue|deny|denied|denies|out\s*of\s*time|trump|administration|executive|\beo\b|immigration|visa|uscis|dhs|border|asylum|refugee|deportation|removal|immigration\s*law|immigration\s*attorney|immigration\s*lawyer|immigration\s*help|immigration\s*advice)"

    subreddit_to_label = {
        # 1. Student Visa Stage
        "f1visa": "general_immigration",
        "opt": "general_immigration", 
        "stemopt": "general_immigration",
        "immigration": "general_immigration",
        "visa": "general_immigration",
        "immigrationusa": "general_immigration",
        "uscisquestions": "general_immigration",
        
        # 2. Work Visa Stage
        "h1b": "general_immigration",
        "workvisas": "general_immigration",
        "immigrationlaw": "general_immigration",
        "immigrationattorney": "general_immigration",
        
        # 3. Permanent Residency Stage
        "greencard": "green_card",
        "greencardprocess": "green_card",
        "uscis": "general_immigration",
        
        # 4. Citizenship Stage
        "citizenship": "general_immigration",
        
        # 5. General Immigration & Legal Issues
        "immigrationquestions": "general_immigration",
        "visasupport": "general_immigration",
        "immigrationnews": "general_immigration",
    }

    def assign_label(text: str, subreddit: str) -> str:
        sub = (subreddit or "").lower()
        if sub in subreddit_to_label:
            return subreddit_to_label[sub]
        if re.search(student_pat, text): return "general_immigration"
        if re.search(work_pat, text): return "general_immigration"
        if re.search(perm_pat, text): return "green_card"
        if re.search(cit_pat, text): return "general_immigration"
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
