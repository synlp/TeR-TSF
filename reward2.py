import ahocorasick

def calculate_normalized_scores(vocab_list, text_list):
    if not vocab_list:
        return [0.0] * len(text_list)

    vocab_list = [word.strip().lower() for word in vocab_list]
    vocab_list = [word for word in vocab_list if word]
    

    automaton = ahocorasick.Automaton()
    unique_words = set()
    for word in vocab_list:

        if word not in unique_words:
            automaton.add_word(word, word)
            unique_words.add(word)
    
    automaton.make_automaton()
    total_vocab_count = len(vocab_list)
    

    scores = []
    for text in text_list:
        text = text.lower()
        matched_words = set()
        

        for _, word in automaton.iter(text):
            matched_words.add(word)
        

        score = len(matched_words) / total_vocab_count
        scores.append(score)
    
    return scores



vocab = ["apple", "banana", "orange", "Apple"]
texts = [
    "I have an Apple and a banana",
    "Oranges are delicious",
    "No fruits here",
    "apple pie, banana split, orange juice"
]
scores = calculate_normalized_scores(vocab, texts)
print(scores)