#!/usr/bin/env python3
"""Build a corpus-derived term thesaurus from paper texts.

Extracts domain terms via title-case phrase detection, acronym expansion,
and high-IDF bigrams. Scores term pairs by sentence co-occurrence weighted
by corpus rarity. Writes data/term_thesaurus.tsv as from→to substitution pairs.

Strategy:
  Phase A — extract candidate terms from every sentence in every paper.
  Phase B — score all term pairs by co-occurrence × inverse document frequency.
  Phase C — for each claim/topic term, pick top co-occurring terms as mappings.
"""

import os, sys, re, math, collections

PAPER_DIR = "paper/sources/text"
MANIFEST = "paper/sources/manifest.tsv"
OUTPUT = "data/term_thesaurus.tsv"
EXISTING = "data/term_thesaurus.tsv"

# Topic descriptions (from nn_bibliography.c DEFAULT_TOPICS)
TOPICS = [
    "verifier gated reasoning training reward model step by step correctness",
    "holographic memory hyperdimensional computing HDC vector symbolic architecture VSA binding retrieval superposition distribute represent",
    "formal methods synthesis grammar verification proof counterexample",
    "neurosymbolic scoring ranking program generation learned proposer",
    "machine learning accountability reproducibility cards datasheets audit",
    "reinforcement learning policy gradient preference optimization rlhf",
    "code generation formal theorem solver search alphageometry sketch",
    "calibration evaluation benchmark generalization measurement ood",
]

# Claim texts + domains (from bibliography_vsa.c)
CLAIMS = [
    "verified gate filter improves top one ranking over raw baseline",
    "shared scorer beats single task scorer on out of domain ranking",
    "holographic memory achieves exact retrieval on isolated pairs",
    "rejected candidate feedback loop improves proposal quality",
    "claim compiler mechanically checks every paper assertion",
    "modular residue filter proves zero false rejections",
    "cross domain transfer benefits from shared hypothesis library",
    "trace determinism audit catches replay sensitive claim drift",
]
DOMAINS = [
    "prime_gap verification ranking gate filter arithmetic",
    "multitask shared scorer generalization cross domain transfer",
    "holographic memory VSA retrieval storage binding",
    "rejected candidate learning feedback proposal loop",
    "claim compiler paper evidence assertion mechanical check",
    "modular arithmetic residue filter certificate proof",
    "cross domain transfer hypothesis library shared",
    "determinism audit trace replay reproducibility",
]

# ---- Phase A: term extraction ----

STOP_WORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "are", "was",
    "not", "but", "they", "has", "had", "were", "can", "its", "all", "been",
    "one", "will", "would", "which", "their", "also", "when", "more", "some",
    "than", "other", "into", "such", "about", "these", "only", "then", "them",
    "over", "each", "any", "may", "could", "should", "between", "after", "most",
    "those", "both", "being", "under", "while", "during", "without", "through",
    "where", "because", "however", "therefore", "thus", "although", "though",
    "using", "used", "well", "within", "often", "based", "figure", "table",
    "section", "paper", "proposed", "approach", "method", "results", "model",
    "data", "set", "use", "two", "three", "many", "new", "first", "different",
    "case", "since", "example", "shown", "given", "following", "et", "al",
}

def sentences(text):
    """Split text into sentences."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 20]

def extract_title_case_phrases(sent):
    """Extract consecutive capitalized-word phrases (2+ words starting with uppercase)."""
    words = sent.split()
    phrases = []
    i = 0
    while i < len(words):
        w = words[i].strip("(),;:\"'")
        if w and w[0].isupper() and len(w) > 1 and w.lower() not in STOP_WORDS:
            j = i + 1
            while j < len(words):
                nw = words[j].strip("(),;:\"'")
                if nw and nw[0].isupper() and len(nw) > 1 and nw.lower() not in STOP_WORDS:
                    j += 1
                else:
                    break
            if j - i >= 2:
                phrase = " ".join(words[i:j]).strip("(),;:\"'")
                phrase = re.sub(r'[^a-zA-Z\s-]', '', phrase).strip()
                if len(phrase) > 0:
                    phrases.append(phrase.lower())
            i = j
        else:
            i += 1
    return phrases

def extract_acronym_pairs(sent):
    """Extract acronym (ALL_CAPS) and nearby definition pairs."""
    pairs = []
    words = sent.split()
    for i, w in enumerate(words):
        w_clean = w.strip("(),;:\"'")
        if re.match(r'^[A-Z]{2,6}$', w_clean):
            acro = w_clean.lower()
            # Look for definition in next 10 words or in parentheses
            context = " ".join(words[max(0,i-10):i+10])
            # Look for parenthetical definition: "VSA (Vector Symbolic Architecture)"
            paren_match = re.search(r'\(([^)]+)\)', context)
            if paren_match:
                defn = paren_match.group(1).strip().lower()
                if len(defn) > 10 and defn[0] == acro[0]:
                    pairs.append((acro, defn))
                    pairs.append((defn, acro))
    return pairs

def extract_terms(text):
    """Extract all candidate terms from a paper text."""
    terms = collections.Counter()
    for sent in sentences(text):
        # Title-case phrases
        for p in extract_title_case_phrases(sent):
            terms[p] += 1
        # Acronym pairs
        for a, d in extract_acronym_pairs(sent):
            terms[a] += 1
            terms[d] += 1
        # High-signal bigrams (both words non-stop, non-trivial)
        words = [w.strip("(),;:\"'").lower() for w in sent.split()]
        words = [w for w in words if len(w) > 2 and w not in STOP_WORDS and w.isalpha()]
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            terms[bigram] += 1
    return terms

# ---- Phase B: co-occurrence scoring ----

def load_papers():
    """Load all paper texts and their topic assignments."""
    papers = []
    # Read manifest
    paper_ids = []
    with open(MANIFEST) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                paper_ids.append(parts[0])
    
    for pid in paper_ids:
        path = os.path.join(PAPER_DIR, f"{pid}.txt")
        if os.path.exists(path):
            with open(path) as f:
                text = f.read()
            papers.append((pid, text))
    return papers

def build_cooccurrence(papers):
    """Build term→sentence index and co-occurrence matrix."""
    # term → set of (paper_idx, sent_idx)
    term_sents = collections.defaultdict(set)
    # paper → set of terms
    paper_terms = collections.defaultdict(set)
    
    for pi, (pid, text) in enumerate(papers):
        sents_list = sentences(text)
        for si, sent in enumerate(sents_list):
            terms = set()
            for p in extract_title_case_phrases(sent):
                terms.add(p)
            for a, d in extract_acronym_pairs(sent):
                terms.add(a); terms.add(d)
            words = [w.strip("(),;:\"'").lower() for w in sent.split()]
            words = [w for w in words if len(w) > 2 and w not in STOP_WORDS and w.isalpha()]
            for i in range(len(words) - 1):
                terms.add(f"{words[i]} {words[i+1]}")
            for t in terms:
                term_sents[t].add((pi, si))
                paper_terms[pi].add(t)
    
    # Compute IDF per term (in how many papers does it appear)
    N = len(papers)
    term_idf = {}
    for t, sents in term_sents.items():
        npapers = len(set(pi for pi, _ in sents))
        term_idf[t] = math.log(N / max(npapers, 1))
    
    # Score term pairs by co-occurrence
    pairs = []
    terms_list = list(term_sents.keys())
    # Only score terms that appear in 2-20 papers (not too common, not too rare)
    scored_terms = {t for t in terms_list if 2 <= len(set(pi for pi, _ in term_sents[t])) <= 20}
    
    for i, t1 in enumerate(terms_list):
        if t1 not in scored_terms: continue
        sents1 = term_sents[t1]
        npapers1 = len(set(pi for pi, _ in sents1))
        for t2 in terms_list[i+1:]:
            if t2 not in scored_terms: continue
            sents2 = term_sents[t2]
            # Co-occurring sentences
            co_sents = sents1 & sents2
            if len(co_sents) < 2:
                continue
            co_papers = len(set(pi for pi, _ in co_sents))
            if co_papers < 2:
                continue
            # Score: co-occurrence rate × shared-paper IDF bonus
            co_rate = len(co_sents) / (len(sents1) + len(sents2))
            idf_bonus = math.log(N / max(co_papers, 1))
            score = co_rate * idf_bonus
            if score > 0.001:
                pairs.append((score, t1, t2))
    
    pairs.sort(reverse=True)
    return pairs, term_sents, term_idf

# ---- Phase C: mapping generation ----

def collect_source_terms():
    """Collect claim and domain terms as source vocabulary."""
    sources = set()
    for text in TOPICS + CLAIMS + DOMAINS:
        words = text.lower().split()
        # Unigrams
        for w in words:
            if len(w) > 2 and w not in STOP_WORDS:
                sources.add(w)
        # Bigrams
        for i in range(len(words) - 1):
            sources.add(f"{words[i]} {words[i+1]}")
        # Triggers
        for i in range(len(words) - 2):
            sources.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    return sources

def build_thesaurus(pairs, term_sents, term_idf):
    """Build from→to mappings using co-occurrence pairs."""
    sources = collect_source_terms()
    
    # For each co-occurrence pair, if one term matches a source, create mapping
    mappings = collections.defaultdict(list)
    
    for score, t1, t2 in pairs:
        t1_match = t1 in sources or any(t1.startswith(s) or s.startswith(t1) for s in sources if len(s) > 3)
        t2_match = t2 in sources or any(t2.startswith(s) or s.startswith(t2) for s in sources if len(s) > 3)
        
        if t1_match and not t2_match:
            mappings[t1].append((score, t2))
        elif t2_match and not t1_match:
            mappings[t2].append((score, t1))
    
    # Keep top-3 per source term
    thesaurus = []
    seen = set()
    for src, candidates in mappings.items():
        candidates.sort(reverse=True)
        for score, target in candidates[:3]:
            pair = (src, target)
            if pair not in seen and src != target:
                thesaurus.append((src, target, score))
                seen.add(pair)
    
    thesaurus.sort(key=lambda x: -x[2])
    return thesaurus

# ---- Main ----

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Phase A: extracting terms from papers...")
    papers = load_papers()
    print(f"  Loaded {len(papers)} papers")
    
    print("Phase B: scoring co-occurrence pairs...")
    pairs, term_sents, term_idf = build_cooccurrence(papers)
    print(f"  {len(pairs)} scored pairs (threshold > 0.001)")
    if pairs:
        print(f"  Top pair: {pairs[0][2]} ↔ {pairs[0][1]} (score={pairs[0][0]:.4f})")
    
    print("Phase C: building thesaurus mappings...")
    thesaurus = build_thesaurus(pairs, term_sents, term_idf)
    print(f"  {len(thesaurus)} generated mappings")
    
    # Merge with existing hand-written pairs (keep them as base)
    existing = []
    if os.path.exists(EXISTING):
        with open(EXISTING) as f:
            for line in f:
                line = line.strip()
                if '\t' in line:
                    existing.append(tuple(line.split('\t')))
    
    # Write: existing pairs first, then generated pairs (no duplicates)
    written = set()
    with open(OUTPUT, 'w') as f:
        for src, tgt in existing:
            f.write(f"{src}\t{tgt}\n")
            written.add((src, tgt))
        for src, tgt, score in thesaurus:
            if (src, tgt) not in written and len(src) > 2 and len(tgt) > 2:
                f.write(f"{src}\t{tgt}\n")
                written.add((src, tgt))
    
    total = len(written)
    print(f"\nWrote {total} pairs to {OUTPUT}")
    print(f"  Existing: {len(existing)}  Generated: {total - len(existing)}")
    
    # Show some generated pairs for key terms
    print("\nSample generated mappings:")
    key_terms = ["holographic", "memory", "retrieval", "binding", "gate", "verifier", "transfer", "determinism"]
    for term in key_terms:
        matches = [(s, t) for s, t, _ in thesaurus if term in s]
        if matches:
            for s, t in matches[:3]:
                print(f"  {s} → {t}")

if __name__ == "__main__":
    main()
