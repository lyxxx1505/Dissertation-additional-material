import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud

# ---------------- NLTK ----------------
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# Download necessary resources
NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}
for pkg, path in NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

# ---------------- VADER ----------------
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- TF-IDF ----------------
from sklearn.feature_extraction.text import TfidfVectorizer


# Input and output
INPUT_CSV = "guardian_esg_merged_dedup.csv"

BASE_DIR = os.path.dirname(os.path.abspath(INPUT_CSV)) or os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_vader_tfidf_layout_en")

TEXT_COLUMNS_CANDIDATES = ["bodyText", "text", "content", "article", "clean_text"]
TEXT_COL = None
NEUTRAL_THRESH = 0.25         
TOP_K_WORDS = 15
FIG_DPI = 200
RANDOM_SEED = 42
MAX_FEATURES = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== Utility Functions =====================
def detect_text_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lengths = {c: df[c].astype(str).str.len().sum()
               for c in df.columns if df[c].dtype == "object"}
    if lengths:
        return max(lengths, key=lengths.get)
    raise ValueError("Unable to automatically detect the text column, please set TEXT_COL manually.")


URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")

def clean_text(s: str) -> str:
    s = s if isinstance(s, str) else ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = NUM_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def nltk_pos_to_wordnet(tag: str) -> str:
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

# Stopwords
stop_en = set(stopwords.words("english"))
custom_stop = {
    "said","say","says","would","could","also","one","two","first","last",
    "esg","environmental","social","governance", "year"  
}
STOP_ALL = stop_en.union(custom_stop)

lemmatizer = WordNetLemmatizer()

def tokenize_lemmatize(text: str):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(w, nltk_pos_to_wordnet(pos)) for (w, pos) in tagged]
    lemmas = [w for w in lemmas if w not in STOP_ALL and len(w) > 2]
    return lemmas

def classify_vader(compound: float, neutral_thresh: float = NEUTRAL_THRESH) -> str:
    if compound >= neutral_thresh:
        return "Positive"
    elif compound <= -neutral_thresh:
        return "Negative"
    else:
        return "Neutral"


# ===================== read and preprocessing =====================
df = pd.read_csv(INPUT_CSV, encoding="utf-8", low_memory=False)
if TEXT_COL is None:
    TEXT_COL = detect_text_column(df, TEXT_COLUMNS_CANDIDATES)

df = df[~df[TEXT_COL].isna()].copy()
df.reset_index(drop=True, inplace=True)

df["text_raw"] = df[TEXT_COL].astype(str)             
df["text_clean"] = df["text_raw"].apply(clean_text)   


# ===================== VADER sentiment analysis =====================
analyzer = SentimentIntensityAnalyzer()
scores = df["text_raw"].apply(analyzer.polarity_scores)

df["vader_neg"] = scores.apply(lambda d: d["neg"])
df["vader_neu"] = scores.apply(lambda d: d["neu"])
df["vader_pos"] = scores.apply(lambda d: d["pos"])
df["vader_compound"] = scores.apply(lambda d: d["compound"])
df["sentiment"] = df["vader_compound"].apply(lambda c: classify_vader(c))

out_csv = os.path.join(OUTPUT_DIR, "sentiment_results_vader_tfidf.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")


# ===================== TF-IDF  =====================
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_lemmatize,
    preprocessor=lambda x: x,
    lowercase=False,
    max_features=MAX_FEATURES,
)
vectorizer.fit(df["text_clean"])


# ===================== Calculate Top-K Important Words for Three Sentiments =====================
labels_order = ["Positive", "Neutral", "Negative"]
label_to_top = {}

for label in labels_order:
    texts = df.loc[df["sentiment"] == label, "text_clean"].tolist()
    if not texts:
        label_to_top[label] = {}
        continue

    tfidf_mat = vectorizer.transform(texts)
    mean_scores = tfidf_mat.mean(axis=0).A1
    vocab = vectorizer.get_feature_names_out()

    top_pairs = sorted(zip(vocab, mean_scores), key=lambda x: x[1], reverse=True)[:TOP_K_WORDS]
    label_to_top[label] = dict(top_pairs)


# ===================== Visualization: Left Pie Chart + Right Three Word Clouds (Beautified) =====================
sent_counts = df["sentiment"].value_counts()
sent_percent = (sent_counts / sent_counts.sum() * 100).round(2)

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(nrows=3, ncols=2, figure=fig, width_ratios=[1.0, 2.2], height_ratios=[1,1,1])

# Left Column: Pie Chart (Shadow, Explosion, Outline, Proportional)
ax_pie = fig.add_subplot(gs[:, 0])
labels_disp = ["Positive", "Neutral", "Negative"]
sizes = [sent_percent.get(k, 0) for k in labels_disp]
explode = (0.03, 0.03, 0.03)
wedges, texts, autotexts = ax_pie.pie(
    sizes,
    labels=[f"{lab} ({val:.2f}%)" for lab, val in zip(labels_disp, sizes)],
    autopct="%1.1f%%",
    startangle=120,
    shadow=True,
    explode=explode,
    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    textprops={"fontsize": 30}
)
ax_pie.axis("equal")

# Right Column Three Rows: Word Clouds for Three Sentiments
ax_pos = fig.add_subplot(gs[0, 1])
ax_neu = fig.add_subplot(gs[1, 1])
ax_neg = fig.add_subplot(gs[2, 1])
axes_map = {"Positive": ax_pos, "Neutral": ax_neu, "Negative": ax_neg}

for label in labels_order:
    ax = axes_map[label]
    freq_dict = label_to_top.get(label, {})
    if not freq_dict:
        ax.text(0.5, 0.5, f"No tokens for {label}", ha="center", va="center", fontsize=14)
        ax.axis("off")
        ax.set_title(label, fontsize=26, fontweight="bold")
        continue

    wc = WordCloud(
        width=1300,
        height=600,
        background_color="white",
        prefer_horizontal=0.95,
        random_state=RANDOM_SEED,
        max_font_size=130,
        min_font_size=12,
        relative_scaling=0.35
    ).generate_from_frequencies(freq_dict)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{label}", fontsize=26, fontweight="bold")

fig.suptitle("Sentiment Pie + TF-IDF Word Clouds", fontsize=20, y=0.98)
plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, f"pie_left_wordclouds_right_top{TOP_K_WORDS}_beautified.png")
plt.savefig(out_fig, dpi=FIG_DPI)
plt.close()

print(f"[OK] Sentiment per line result: {out_csv}")
print(f"[OK] Merged beautified visualization: {out_fig}")


# ===================== NEW: Read and Merge Topic Mapping, Replicate Visualization by Topic2/Non-Topic2 =====================
# Explanation: Using already calculated df['sentiment'] and fitted vectorizer,
#              for two subsets (topic==2 and topic!=2), create "pie chart + three word clouds (Top-15)".
# ----------------------------------------------------------------------------------
# === NEW: Read the topic mapping file
TOPIC_MAP_CSV = os.path.join(BASE_DIR, "topic_assignments.csv")
if os.path.exists(TOPIC_MAP_CSV):
    topic_map = pd.read_csv(TOPIC_MAP_CSV, encoding="utf-8")
    if "url" in df.columns and "url" in topic_map.columns and "topic" in topic_map.columns:
        dfm = df.merge(topic_map[["url", "topic"]], on="url", how="inner")
    else:
        # If the original df doesn't have the 'url' column, fall back to the original df (to avoid errors)
        print("[WARN] Could not find 'url' in data or missing 'topic' in mapping table, skipping merge.")
        dfm = df.copy()
else:
    print(f"[WARN] {TOPIC_MAP_CSV} not found, skipping topic subset visualization.")
    dfm = df.copy()

# === NEW: Split into topic2 and non-topic2 subsets
df_topic2 = dfm[dfm.get("topic", -1) == 2].copy()
df_not2   = dfm[dfm.get("topic", -1) != 2].copy()

# === NEW: Reuse the same visualization (function-based)
def visualize_subset(df_sub: pd.DataFrame, tag: str):
    
    if df_sub.empty:
        print(f"[WARN] Subset {tag} is empty, skipping visualization.")
        return None

    # Pie chart data
    sent_counts_sub = df_sub["sentiment"].value_counts()
    sent_percent_sub = (sent_counts_sub / sent_counts_sub.sum() * 100).round(2)

    # Top-K for each sentiment
    label_to_top_sub = {}
    for label in labels_order:
        texts = df_sub.loc[df_sub["sentiment"] == label, "text_clean"].tolist()
        if not texts:
            label_to_top_sub[label] = {}
            continue
        tfidf_mat = vectorizer.transform(texts)
        mean_scores = tfidf_mat.mean(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        top_pairs = sorted(zip(vocab, mean_scores), key=lambda x: x[1], reverse=True)[:TOP_K_WORDS]
        label_to_top_sub[label] = dict(top_pairs)

    # Create the plot: identical layout and style
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(nrows=3, ncols=2, figure=fig, width_ratios=[1.0, 2.2], height_ratios=[1,1,1])

    # Left: Pie chart
    ax_pie = fig.add_subplot(gs[:, 0])
    sizes = [sent_percent_sub.get(k, 0) for k in labels_disp]
    explode = (0.03, 0.03, 0.03)
    ax_pie.pie(
        sizes,
        labels=[f"{lab} ({val:.2f}%)" for lab, val in zip(labels_disp, sizes)],
        autopct="%1.1f%%",
        startangle=120,
        shadow=True,
        explode=explode,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 30}
    )
    ax_pie.axis("equal")

    # Right: Three word clouds
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_neu = fig.add_subplot(gs[1, 1])
    ax_neg = fig.add_subplot(gs[2, 1])
    axes_map = {"Positive": ax_pos, "Neutral": ax_neu, "Negative": ax_neg}

    for label in labels_order:
        ax = axes_map[label]
        freq_dict = label_to_top_sub.get(label, {})
        if not freq_dict:
            ax.text(0.5, 0.5, f"No tokens for {label}", ha="center", va="center", fontsize=14)
            ax.axis("off")
            ax.set_title(label, fontsize=26, fontweight="bold")
            continue

        wc = WordCloud(
            width=1300,
            height=600,
            background_color="white",
            prefer_horizontal=0.95,
            random_state=RANDOM_SEED,
            max_font_size=130,
            min_font_size=12,
            relative_scaling=0.35
        ).generate_from_frequencies(freq_dict)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{label}", fontsize=26, fontweight="bold")

    fig.suptitle(f"[{tag}] Sentiment Pie + TF-IDF Word Clouds",
                 fontsize=20, y=0.98)
    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR, f"subset_{tag}_pie_wordclouds_right_top{TOP_K_WORDS}_beautified.png"
    )
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()
    return out_path

# === NEW: Export subset results (for further statistics/time series)
if not df_topic2.empty:
    df_topic2.to_csv(os.path.join(OUTPUT_DIR, "sentiment_results_topic2.csv"),
                     index=False, encoding="utf-8-sig")
if not df_not2.empty:
    df_not2.to_csv(os.path.join(OUTPUT_DIR, "sentiment_results_non_topic2.csv"),
                   index=False, encoding="utf-8-sig")

# === NEW: Generate visualizations for both subsets
p1 = visualize_subset(df_topic2, "topic2")
p2 = visualize_subset(df_not2, "non_topic2")
if p1: print(f"[OK] Subset visualization (topic2): {p1}")
if p2: print(f"[OK] Subset visualization (non-topic2): {p2}")
print(dfm.columns)  # Check all column names