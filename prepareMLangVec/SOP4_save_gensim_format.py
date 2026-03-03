from gensim.models.fasttext import load_facebook_vectors

# 1. Load the slow way one last time
print("Loading FB binary... (this will take a few minutes)")
wv = load_facebook_vectors('cjk_english_300.bin')

# 2. Save it in Gensim's high-speed native format
# This will create multiple files (e.g., .kv, .kv.vectors_ngrams.npy)
wv.save('cjk_fasttext.kv')
print("Saved as native Gensim KeyedVectors!")
