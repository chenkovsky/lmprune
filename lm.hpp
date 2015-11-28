#ifndef __LM_HPP__
#define __LM_HPP__ 1
#include <iostream>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <string>
using namespace std;

#include "docopt.cpp/docopt.h"
#include "gram.hpp"
#define DONT_PRUNE 2
#define PRUNED 1
typedef const string *const_str_p;
static const string bos = "<s>";
static const string eos = "</s>";
static const prob_t LogP_Zero = -HUGE_VAL; /* log(0) */
static const prob_t LogP_Inf = HUGE_VAL;            /* log(Inf) */
static const prob_t LogP_One = 0.0;  /* log(1) */
static const prob_t Prob_Epsilon = 3e-06;
static const prob_t min_log_prob = -99;
static inline double exp10(double d) {
    if (d == LogP_Zero) {
        return 0.0;
    }
    return exp(d * M_LN10);
}

namespace std {
template<> struct hash<GramNode *> {
    hash<word_id_t> id_hash;
    size_t operator ()(GramNode *const k) const {
        size_t ret = 0;
        const GramNode *k_pt = k;
        while (k_pt) {
            ret ^= id_hash(k_pt->word_id);
            k_pt = k_pt->prefix;
        }
        return ret;
    }
};
}

struct GramNodeEqual {
    bool operator ()(GramNode const *a, GramNode const *b) const {
        return *a == *b;
    }
};

typedef unordered_set<GramNode *, std::hash<GramNode *>, GramNodeEqual> GramNodeSet;

class LanguageModel {
    size_t ngram_num_per_order[MAX_ORDER]; //模型载入时大小
    size_t used_ngram_num_per_order[MAX_ORDER]; //模型实际大小
    GramNode *grams_per_order[MAX_ORDER];
    uint32_t order;                             //模型order
    unordered_map<string, word_id_t> word2id;   //单词到word_id的映射
    GramNodeSet node_set; //gram的集合
    const_str_p *wordid2str;   // word_id到word的映射
    size_t total_gram_num; //载入时总共gram数
    size_t low_gram_num;
    size_t bos_gram_id;
    size_t vocab_size;
    size_t used_gram_num;

    GramNode *grams_buff;
    prob_t *prob_buff;
    prob_t *bow_buff;
    GramNode **child_buff;
    size_t *child_num_buff;
    GramNode **sib_buff;
    uint8_t *mask_for_grams;
    GramNode node_buff[MAX_ORDER]; //这个buff用来查询真正的node

    static const string ngram_prefix;
    static const string data_prefix;
    static const string ngram_block_prefix;
    static const string ngram_block_suffix;
    static const string end_prefix;
    static const string sep;
    static const string escape;


    LanguageModel(size_t *ngram_num_per_order_, uint32_t order);
    LanguageModel() {
    }
    word_id_t word2Idx(const string &str);
    word_id_t word2Idx_append_if_not_exist(const string &str);
    void add_gram(word_id_t *words, uint32_t order, prob_t prob, prob_t bow);
    void load_finish(); //当加载完调用。会做一些检查。
    template<class T> T& puts_gram(T &os, const GramNode *n) const {
        word_id_t word_idx[MAX_ORDER];
        uint32_t order = n->words(word_idx);
        for (int32_t i = order-1; i > 0; i--) {
            os << (*wordid2str[word_idx[i]]) << " ";
        }
        os << (*wordid2str[word_idx[0]]);
        return os;
    }

    template<class T> T& puts_gram(T &os, word_id_t* word_idx, uint32_t order) const {
        for (int32_t i = 0; i + 1 < order; i++) {
            os << (*wordid2str[word_idx[i]]) << " ";
        }
        os << (*wordid2str[word_idx[order-1]]);
        return os;
    }

    double calc_distance(uint32_t lvl, GramNode *n);
    void recalc_bow(); 
    void recalc_bow_for_node(GramNode &node, uint32_t lvl);
    void prune_level(uint32_t lvl, size_t cut_num);
    //prob_t get_pr(GramNode *nodes, uint32_t order);
    bool computeBOW(size_t gram_id, word_id_t *context,uint32_t lvl, double &numerator, double& denominator);
    prob_t wordProbBO(word_id_t *word_ids, uint32_t context_len);

public:
    ~LanguageModel();
    template<class T> static LanguageModel* load(T &in);
    static LanguageModel* load(const char *filename);
    template<class T> void load_important_gram(T &in);
    void load_important_gram(const char* filename);
    void save(const char *filename);
    const GramNode* gram(const word_id_t *word_ids, uint32_t order);
    const GramNode* gram(const vector<string> & vec);
    inline prob_t prob(size_t gram_id) {
        return prob_buff[gram_id];
    }
    inline prob_t bow(size_t gram_id) {
        return bow_buff[gram_id];
    }
    template<class T> friend T&operator<<(T &os, const LanguageModel &lm);
    uint32_t get_order() {
        return order;
    }
    void prune(bool cut, uint32_t nlvl, size_t *ncut);
};


#endif

