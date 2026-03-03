import fasttext

def extract_top_vectors(bin_path, out_vec_path, limit=200000):
    model = fasttext.load_model(bin_path)
    
    # FIX: Add on_unicode_error='replace' or 'ignore'
    # This tells the C++ backend not to crash on invalid UTF-8 sequences
    words = model.get_words(include_freq=False, on_unicode_error='replace')
    
    with open(out_vec_path, 'w', encoding='utf-8') as f:
        # Use min() in case the model has fewer words than your limit
        actual_limit = min(limit, len(words))
        f.write(f"{actual_limit} 300\n")
        
        for i in range(actual_limit):
            word = words[i]
            vec = model.get_word_vector(word)
            vec_str = " ".join([f"{x:.6f}" for x in vec])
            f.write(f"{word} {vec_str}\n")
    print(f"Successfully extracted {actual_limit} vectors to {out_vec_path}")

# Run this for all three (this will take some time and RAM)
extract_top_vectors('cc.en.300.bin', 'en.vec')
extract_top_vectors('cc.zh.300.bin', 'zh.vec')
extract_top_vectors('cc.ja.300.bin', 'ja.vec')
