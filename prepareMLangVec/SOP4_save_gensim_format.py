from gensim.models import KeyedVectors

# 1. Load the Word2Vec binary file (Corrected from SOP3 output)
print("Loading Word2Vec binary format...")
# This will be MUCH faster and won't throw unicode errors
wv = KeyedVectors.load_word2vec_format('cjk_english_300.bin', binary=True)

# 2. Save it in Gensim's native format for mmap loading
print("Saving as native Gensim KeyedVectors...")
wv.save('cjk_fasttext.kv')

print("Success! You can now load it instantly with:")
print("ft_kv = KeyedVectors.load('cjk_fasttext.kv', mmap='r')")
