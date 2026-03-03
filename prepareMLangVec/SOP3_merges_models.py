import numpy as np
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes
from deep_translator import GoogleTranslator
from tqdm import tqdm # For progress bar

def generate_seed_dictionary(en_model, target_vec_path, lang_code, limit=3000):
    """
    en_model: Your loaded English KeyedVectors
    target_vec_path: Path to 'ja.vec' or 'zh.vec' to verify word existence
    lang_code: 'ja' or 'zh-TW'
    """
    print(f"Loading target vocabulary from {target_vec_path}...")
    # Just load the words from the .vec file to save RAM
    with open(target_vec_path, 'r', encoding='utf-8') as f:
        target_vocab = {line.split()[0] for i, line in enumerate(f) if i > 0}

    translator = GoogleTranslator(source='en', target=lang_code)

    # 1. Select candidates: Skip the first 100 words (mostly symbols/stopwords)
    # We take words from index 100 to 5000 to find 3000 valid pairs
    candidates = en_model.index_to_key[100:5000]

    seed_dictionary = []
    print(f"Generating {lang_code} seed dictionary...")

    for word in tqdm(candidates):
        if len(seed_dictionary) >= limit:
            break
        try:
            # Translate and clean
            translated = translator.translate(word).strip()

            # 2. Critical Check: Does the translated word exist in our target model?
            if translated in target_vocab:
                seed_dictionary.append((translated, word)) # (Source, Target)
        except Exception:
            continue

    return seed_dictionary

def align_and_merge(base_model, source_vec_path, dictionary_pairs):
    """
    base_model: The English KeyedVectors (already loaded)
    source_vec_path: Path to 'ja.vec' or 'zh.vec'
    dictionary_pairs: List of (source_word, target_word) e.g., [("猫", "cat")]
    """
    print(f"Loading {source_vec_path}...")
    source_model = KeyedVectors.load_word2vec_format(source_vec_path, binary=False)
    
    # 1. Build the training matrices for Procrustes
    src_matrix, tgt_matrix = [], []
    for src_w, tgt_w in dictionary_pairs:
        if src_w in source_model and tgt_w in base_model:
            src_matrix.append(source_model[src_w])
            tgt_matrix.append(base_model[tgt_w])
    
    # 2. Calculate the Rotation Matrix (R)
    R, _ = orthogonal_procrustes(np.array(src_matrix), np.array(tgt_matrix))
    
    # 3. Apply rotation to ALL vectors in the source model
    # Note: .vectors is the raw numpy array in KeyedVectors
    source_model.vectors = source_model.vectors @ R
    
    return source_model

# --- MAIN EXECUTION ---

# 1. Load English as the Anchor
print("Loading English Anchor...")
w2v_multilingual = KeyedVectors.load_word2vec_format('en.vec', binary=False, limit=200000)

# 2. Get your 3000-word seed dictionaries
# ja_en_pairs = [("猫", "cat"), ("犬", "dog"), ...]
# zh_en_pairs = [("貓", "cat"), ("狗", "dog"), ...]
ja_en_pairs = generate_seed_dictionary(w2v_multilingual, 'ja.vec', 'ja', limit=3000)
zh_en_pairs = generate_seed_dictionary(w2v_multilingual, 'zh.vec', 'zh-TW', limit=3000)


# 3. Align and Add Japanese
ja_aligned = align_and_merge(w2v_multilingual, 'ja.vec', ja_en_pairs)
w2v_multilingual.add_vectors(ja_aligned.index_to_key, ja_aligned.vectors)

# 4. Align and Add Chinese
zh_aligned = align_and_merge(w2v_multilingual, 'zh.vec', zh_en_pairs)
w2v_multilingual.add_vectors(zh_aligned.index_to_key, zh_aligned.vectors)

# ... after your add_vectors() calls ...

def safely_finalize_and_save(kv_model, output_path):
    print("Cleaning and validating model vectors...")

    # 1. Identify valid indices (where vector is not all zeros or NaNs)
    valid_keys = []
    valid_vectors = []

    for key in kv_model.index_to_key:
        vec = kv_model.get_vector(key)
        # Check if the vector is valid (not all zeros and contains no NaNs)
        if np.any(vec) and not np.isnan(vec).any():
            valid_keys.append(key)
            valid_vectors.append(vec)

    # 2. Re-create a clean KeyedVectors object
    print(f"Creating clean model: {len(valid_keys)} valid words found.")
    clean_kv = KeyedVectors(vector_size=kv_model.vector_size)
    clean_kv.add_vectors(valid_keys, valid_vectors)

    # 3. Safe Normalization
    print("Normalizing vectors...")
    clean_kv.unit_normalize_all()

    # 4. Save using the sort_attr=None fix
    print(f"Saving to {output_path}...")
    clean_kv.save_word2vec_format(
        output_path,
        binary=True,
        sort_attr=None
    )
    return clean_kv

# Execute the safe save
w2v_multilingual = safely_finalize_and_save(w2v_multilingual, 'cjk_english_300.bin')
print("SOP3 Complete.")

