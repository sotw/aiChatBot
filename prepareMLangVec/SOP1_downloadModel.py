import fasttext
import fasttext.util

# This will download the .bin models (approx 4-5GB each)
# They are NOT aligned yet, but this is the first step to get the data
print("Downloading English...")
fasttext.util.download_model('en', if_exists='ignore') 
print("Downloading Traditional Chinese...")
fasttext.util.download_model('zh', if_exists='ignore')
print("Downloading Japanese...")
fasttext.util.download_model('ja', if_exists='ignore')
