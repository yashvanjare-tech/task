import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def summarize_text(text, summary_ratio=0.3):
    
    sentences = sent_tokenize(text)
    if not sentences:
        return text  

    
    stop_words = set(stopwords.words('english'))
    word_frequencies = defaultdict(int)
    for word in word_tokenize(text.lower()):
        if word.isalnum() and word not in stop_words:
            word_frequencies[word] += 1

    if not word_frequencies:
        return text  
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    
    sentence_scores = defaultdict(float)
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        if 5 < len(words) < 30:  
            for word in words:
                if word in word_frequencies:
                    sentence_scores[sentence] += word_frequencies[word]

    
    num_sentences = max(1, int(len(sentences) * summary_ratio))
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
 
    summary_sentences = sorted(ranked_sentences, key=sentences.index)

    return ' '.join(summary_sentences)

def get_user_input():
    """Gets multiline input from user until empty line."""
    print("Enter your text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return '\n'.join(lines)

if __name__ == "__main__":
   
    user_text = get_user_input()

    if not user_text.strip():
        print("Error: No text entered!")
    else:
       
        summary = summarize_text(user_text, summary_ratio=0.3)
        print("\nOriginal Text Length:", len(user_text), "characters")
        print("Summary Length:", len(summary), "characters")
        print("\nSummarized Text:\n", summary)
