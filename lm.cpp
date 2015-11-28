#include "lm.hpp"
LanguageModel::LanguageModel(size_t *ngram_num_per_order_, uint32_t order_){
    assert(order_ >= 1);
    order = order_;
    memcpy(ngram_num_per_order, ngram_num_per_order_, sizeof(size_t)*order);
    memset(used_ngram_num_per_order, 0, sizeof(used_ngram_num_per_order));

    wordid2str = new const_str_p[ngram_num_per_order[0]];

    low_gram_num = 0;
    for (uint32_t i = 0; (i + 1) < order;i++) {
        low_gram_num += ngram_num_per_order[i];
    }
    total_gram_num = low_gram_num + ngram_num_per_order[order-1];

    bos_gram_id = -1;
    vocab_size = 0;
    used_gram_num = 0;
    
    grams_buff = new GramNode[total_gram_num];
    size_t offset = 0;
    for (uint32_t i = 0; i < order; i++) {
        grams_per_order[i] = grams_buff + offset;
        offset += ngram_num_per_order[i];
    }

    prob_buff = new prob_t[total_gram_num];
    bow_buff = new prob_t[low_gram_num];
    child_buff = new GramNode*[low_gram_num]();
    child_num_buff = new size_t[low_gram_num]();
    sib_buff = new GramNode*[total_gram_num]();
    mask_for_grams = new uint8_t[total_gram_num]();

    node_set.reserve(total_gram_num);
    word2id.reserve(ngram_num_per_order[0]);

    memset(node_buff, 0, sizeof(node_buff));
    for (uint32_t i = MAX_ORDER-1; i > 0; i--) {
        node_buff[i].prefix = &node_buff[i-1];
        node_buff[i].gram_id = -1;
    }
    node_buff[0].gram_id = -1;
    node_buff[0].prefix = NULL;
}

LanguageModel::~LanguageModel(){
    delete [] wordid2str;
    delete [] grams_buff;
    delete [] prob_buff;

    delete [] bow_buff;
    delete [] child_buff;

    delete [] child_num_buff;
    delete [] sib_buff;
    delete [] mask_for_grams;
}

word_id_t LanguageModel::word2Idx(const string& str){
    auto found = word2id.find(str);
    if (found == word2id.end()) {
        return -1;
    }
    return found->second;
}

word_id_t LanguageModel::word2Idx_append_if_not_exist(const string& str){
    auto found = word2id.find(str);
    if (found != word2id.end()) {
        return found->second;
    }
    word_id_t ret = vocab_size;
    vocab_size++;
    auto inserted = word2id.insert(make_pair(str, ret));
    wordid2str[ret] = &inserted.first->first;
    return ret;
}


void LanguageModel::add_gram(word_id_t* words, uint32_t cur_order, prob_t prob, prob_t bow){
    used_ngram_num_per_order[cur_order-1]++;
    GramNode& node = grams_buff[used_gram_num];
    node.gram_id = used_gram_num;
    node.word_id = words[cur_order-1];
    node.prefix = NULL;
    if (cur_order >= 2) {//有prefix
        for (uint32_t i = 0; i < cur_order-1; i++) {
            node_buff[i].word_id = words[i];
        }
        auto prefix_ = node_set.find(&node_buff[cur_order - 2]);
        assert(prefix_ != node_set.end());
        node.prefix = (*prefix_);
        sib_buff[used_gram_num] = child_buff[node.prefix->gram_id];
        child_buff[node.prefix->gram_id] = &node;
        child_num_buff[node.prefix->gram_id]++;
    }
    node_set.insert(&node);

    prob_buff[used_gram_num] = prob;
    if (cur_order < order) {
        bow_buff[used_gram_num] = bow;
    }

    used_gram_num++;
}

void LanguageModel::load_finish(){
    size_t used_gram_num_ = 0;
    for (uint32_t i = 0; i < order; i++) {
        assert(used_ngram_num_per_order[i] == ngram_num_per_order[i]);
        used_gram_num_ += used_ngram_num_per_order[i];
    }
    assert(used_gram_num == used_gram_num_);
    assert(vocab_size == used_ngram_num_per_order[0]);

    auto bos_ = word2id.find(bos);
    if(bos_ != word2id.end()){
      //该语言模型中包含bos,那么将bos的概率设为eos的
      word_id_t word_buff [1];
      word_buff[0] = bos_->second;
      const GramNode* bos_gram = gram(word_buff,1);
      assert(bos_gram != NULL);

      bos_gram_id = bos_gram->gram_id;
      auto eos_ = word2id.find(eos);
      assert(eos_ != word2id.end());

      word_buff[0] = eos_->second;
      const GramNode* eos_gram = gram(word_buff, 1);
      assert(eos_gram != NULL);

      prob_buff[bos_gram->gram_id] = prob_buff[eos_gram->gram_id];
    }

}

const GramNode* LanguageModel::gram(const word_id_t* word_ids, uint32_t cur_order){
    for (uint32_t i = 0; i < cur_order; i++) {
        node_buff[i].word_id = word_ids[i];
    }
    auto found = node_set.find(&node_buff[cur_order-1]);
    if (found != node_set.end()) {
        return (*found);
    }
    return NULL;
}

const GramNode* LanguageModel::gram(const vector<string> & vec){
    word_id_t words[MAX_ORDER];
    uint32_t order = vec.size();
    for (uint32_t i = 0; i < order;i++) {
        words[i] = word2Idx(vec[i]);
        if (words[i] == -1) {
            return NULL;
        }
    }
    return gram(words,order);
}




