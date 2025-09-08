import os, re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# ---------------- Parameters ----------------
INPUT_CSV   = "guardian_esg_merged_dedup.csv"
TEXT_COL    = "bodyText"   # change here if column name is different
PASSES      = 10
TOPN_WORDS  = 10           # number of top words per topic
USE_BIGRAM  = True         # True=allow bigram, False=only unigrams

# ---------------- Stopwords ----------------
DEFAULT_STOP = {w.lower() for w in stopwords.words('english')}
CUSTOM_STOP = {
    'said','say','says','would','could','also','one','two','three','first',
    'new','year','years','time','many','may','like','get','us','uk','see',
    'still','well','going','much','today','tomorrow','week','weeks','month',
    'months','day','days','back','mr','mrs','want','way','since','people',
    'need','country','make','support','including','life','last',
    'think','thing','know','made','come','must','already','city',
    'book','look','big','better','part','young','million','cut',
    'talk','needed','number','home','old','around','good',
    'something','feel','change','even','plan','work','take','trump'
}
STOPWORDS = DEFAULT_STOP | {w.lower() for w in CUSTOM_STOP}

# ---------------- Preprocessing function ----------------
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Clean + tokenize + remove stopwords + lemmatize"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)             # remove punctuation
    tokens = []
    for w in word_tokenize(text):
        if w.isalpha():
            w = lemmatizer.lemmatize(w)  # lemmatize
            if w not in STOPWORDS and len(w) > 2:
                tokens.append(w)
    return tokens

# ---------------- Main ----------------
def main():
    # 1. Load data
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"File not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
   
    docs = df[TEXT_COL].dropna().tolist()

    # 2. Preprocess text
    token_lists = [preprocess(doc) for doc in docs]

    # 3. Apply bigram if enabled
    if USE_BIGRAM:
        bigram_phrases = Phrases(token_lists, min_count=10, threshold=10)
        bigram = Phraser(bigram_phrases)
        token_lists = [bigram[doc] for doc in token_lists]

    # 4. Build dictionary and corpus
    dictionary = Dictionary(token_lists)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(toks) for toks in token_lists]

    # 5. Find best K using coherence
    coherences = []
    for k in range(2, 8):
        m = LdaModel(corpus=corpus, id2word=dictionary,
                     num_topics=k, passes=5, random_state=42)
        cm = CoherenceModel(model=m, texts=token_lists,
                            dictionary=dictionary, coherence='c_v')
        coherences.append((k, cm.get_coherence()))

    ks, cs = zip(*coherences)
    plt.plot(ks, cs, marker="o")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence")
    plt.title("Coherence Score vs Number of Topics")
    plt.savefig("coherence.png")
    plt.close()

    # select best K
    best_k = max(coherences, key=lambda x: x[1])[0]
    print(f"\n[INFO] Best number of topics according to coherence: {best_k}")

    # 6. Train final LDA with best K
    lda = LdaModel(corpus=corpus, id2word=dictionary,
                   num_topics=best_k, passes=PASSES, random_state=42)
    lda.save(f"lda_model_k{best_k}.lda")

    print("\n--- LDA Topics ---")
    for t in lda.print_topics(num_words=10):
        print(t)

    # 7. Plot top words per topic
    topics = lda.show_topics(num_topics=best_k, num_words=TOPN_WORDS, formatted=False)
    ncols = 3
    nrows = (best_k + ncols - 1) // ncols
    plt.figure(figsize=(5*ncols, 3.5*nrows))
    for idx, (topic_id, terms) in enumerate(topics, start=1):
        plt.subplot(nrows, ncols, idx)
        words, weights = zip(*terms)
        plt.barh(range(len(words)), weights)
        plt.yticks(range(len(words)), words)
        plt.gca().invert_yaxis()
        plt.title(f"Topic {topic_id}")
    plt.suptitle("LDA Topics (Top-N Words)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"lda_topics_topwords_k{best_k}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 8. pyLDAvis interactive visualization
    vis = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis, f"lda_vis_k{best_k}.html")

    # 9. Quarterly line chart with total news volume and 5 topics per point
    try:
        # (a) Align rows with non-null text in the original df (keep order consistent)
        df_nonnull = df.loc[df[TEXT_COL].notna(), ['url','date']].reset_index(drop=True)
        
        # (b) Compute dominant topic for each document
        def dominant_topic(bow):
            topics_for_doc = lda.get_document_topics(bow, minimum_probability=0.0)
            return max(topics_for_doc, key=lambda x: x[1])[0] if topics_for_doc else None
        
        dom_topics = [dominant_topic(bow) for bow in corpus]

        # Ensure alignment between dom_topics, df_nonnull, and token_lists
        assert len(dom_topics) == len(df_nonnull) == len(token_lists), \
            f"Length mismatch: dom_topics={len(dom_topics)}, df_nonnull={len(df_nonnull)}, token_lists={len(token_lists)}"
        
        # (c) Build document-level metadata: date → quarter; filter time window (2018Q1 to 2025Q2)
        docs_meta = df_nonnull.copy()
        docs_meta['topic'] = dom_topics
        docs_meta['date'] = pd.to_datetime(docs_meta['date'], errors='coerce', utc=True).dt.tz_localize(None)
        start_date = pd.Timestamp('2018-01-01')     # still timezone-naive
        end_date   = pd.Timestamp('2025-06-30')     # still timezone-naive
        docs_meta = docs_meta[(docs_meta['date'] >= start_date) & (docs_meta['date'] <= end_date)]
        docs_meta['quarter'] = docs_meta['date'].dt.to_period('Q')
        
        # (d) Construct full quarterly index (2018Q1 ~ 2025Q2)
        full_q = pd.period_range('2018Q1', '2025Q2', freq='Q')
        
        # (e) Count number of articles per quarter
        vol_q = docs_meta.groupby('quarter').size().reindex(full_q, fill_value=0)
        
        # (f) Count the number of topics per quarter
        topic_counts = docs_meta.groupby(['quarter', 'topic']).size().unstack(fill_value=0)
        topic_counts = topic_counts.reindex(full_q, fill_value=0)

        # (g) Line chart: news volume and topic proportions
        x_dt = full_q.to_timestamp()
        y_ct = vol_q.values
        plt.figure(figsize=(15, 7))

        # Plot total volume of news
        plt.plot(x_dt, y_ct, marker='o', color='black', label='Total News Volume', linewidth=2)

        # Plot topics per quarter (using different colors)
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i in range(5):  # We have 5 topics
            plt.plot(x_dt, topic_counts[i], marker='o', label=f'Topic {i}', color=colors[i], linewidth=2)
    
        # Annotate topic labels and add legend
        plt.title("Quarterly ESG News Volume & Topic Proportions (2018Q1–2025Q2)")
        plt.xlabel("Quarter")
        plt.ylabel("Article Count / Topic Proportion")
        plt.legend(title="Topics")
        plt.tight_layout()
        plt.savefig("quarterly_news_and_topics.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"[WARN] Failed to build annotated quarterly charts: {e}")

    # 11. Export topic assignments (url -> dominant topic) ----------------
    try:
        df_nonnull = df.loc[df[TEXT_COL].notna(), ['url','date']].reset_index(drop=True)

        def dominant_topic(bow):
            topics_for_doc = lda.get_document_topics(bow, minimum_probability=0.0)
            return max(topics_for_doc, key=lambda x: x[1])[0] if topics_for_doc else None

        dom_topics = [dominant_topic(bow) for bow in corpus]

        # 对齐检查
        assert len(dom_topics) == len(df_nonnull), \
            f"Length mismatch: dom_topics={len(dom_topics)} vs df_nonnull={len(df_nonnull)}"

        topic_map = df_nonnull[['url','date']].copy()
        topic_map['topic'] = dom_topics
        topic_map.to_csv("topic_assignments.csv", index=False, encoding="utf-8-sig")

        print("[✔] Exported topic assignments to topic_assignments.csv")
    except Exception as e:
        print(f"[WARN] Failed to export topic assignments: {e}")
        
    print("\n[✔] Generated files:")
    print(" - coherence.png (Coherence curve)")
    print(f" - lda_topics_topwords_k{best_k}.png (Top-N words per topic)")
    print(f" - lda_vis_k{best_k}.html (Interactive visualization)")
    print(" - quarterly_topics_with_proportions.png (Quarterly volume with topics and proportions)")
    print(" - topic_assignments.csv (url → topic mapping)")

if __name__ == "__main__":
    main()
