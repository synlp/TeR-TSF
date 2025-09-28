import ahocorasick

def calculate_normalized_scores(vocab_list, text_list):
    """
    计算文本列表中每个文本的归一化分数
    :param vocab_list: 词汇列表，包含需要检测的单词
    :param text_list: 文本列表，包含待检测的字符串
    :return: 归一化分数列表，每个元素对应text_list中相应文本的分数
    """
    # 处理空词汇表的边界情况
    if not vocab_list:
        return [0.0] * len(text_list)
    
    # 预处理：转换为小写并过滤空字符串
    vocab_list = [word.strip().lower() for word in vocab_list]
    vocab_list = [word for word in vocab_list if word]
    
    # 构建Aho-Corasick自动机
    automaton = ahocorasick.Automaton()
    unique_words = set()
    for word in vocab_list:
        # 只添加唯一的单词到自动机
        if word not in unique_words:
            automaton.add_word(word, word)
            unique_words.add(word)
    
    automaton.make_automaton()
    total_vocab_count = len(vocab_list)  # 原始词汇表长度（含重复）
    
    # 计算每个文本的分数
    scores = []
    for text in text_list:
        text = text.lower()
        matched_words = set()
        
        # 在文本中搜索所有匹配的词汇
        for _, word in automaton.iter(text):
            matched_words.add(word)
        
        # 计算归一化分数
        score = len(matched_words) / total_vocab_count
        scores.append(score)
    
    return scores


# 示例数据
vocab = ["apple", "banana", "orange", "Apple"]  # 注意大小写和重复
texts = [
    "I have an Apple and a banana",
    "Oranges are delicious",
    "No fruits here",
    "apple pie, banana split, orange juice"
]
# 计算分数
scores = calculate_normalized_scores(vocab, texts)
print(scores)  # 输出: [0.5, 0.25, 0.0, 0.75]