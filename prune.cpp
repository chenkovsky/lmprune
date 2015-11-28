#include "lm.hpp"

const prob_t LogP_Zero = -HUGE_VAL; /* log(0) */
const prob_t LogP_Inf = HUGE_VAL;           /* log(Inf) */
const prob_t LogP_One = 0.0;  /* log(1) */
const prob_t Prob_Epsilon = 3e-06;

static const string bos = "<s>";
static const string eos = "</s>";
static const prob_t min_log_prob = -99;
static inline double exp10(double d) {
    if (d == LogP_Zero) {
        return 0.0;
    }
    return exp(d * M_LN10);
}

struct GramNode {
    word_id_t word_id;
    size_t gram_id;
    prob_t prob;
    // h_n,....h_1, w
    // prefix is: h_n,...h_1
    struct GramNode *prefix; // prefix
    bool operator==(const GramNode &r) const {
        if (gram_id != -1 && r.gram_id != -1) {
            return gram_id == r.gram_id;
        }
        //如果能够比较gram_id就比较gram_id,加快速度，如果不行，就比较每个word的id
        const GramNode *o1 = this;
        const GramNode *o2 = &r;
        while (o1 && o2) {
            if (o1->word_id != o2->word_id) {
                return false;
            }
            o1 = o1->prefix;
            o2 = o2->prefix;
        }
        return o1 == NULL && o2 == NULL;
    }
private:
    GramNode(GramNode &o);
};

struct GramInfo {
    GramNode *node;
    double dis; // distance
    bool operator<(const GramInfo &r) const { return dis < r.dis; }
    bool operator==(const GramInfo &r) const { return dis == r.dis; }
};


struct GramInfo {
    GramNode *node;
    double dis; // distance
    bool operator<(const GramInfo &r) const { return dis < r.dis; }
    bool operator==(const GramInfo &r) const { return dis == r.dis; }
};

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

class LanguageModel {
    size_t ngram_num_per_order[MAX_ORDER]; //模型载入时大小
    size_t ngram_num_order_less_than[MAX_ORDER]; // order小于i的所有gram的数目
    size_t used_ngram_num_per_order[MAX_ORDER]; //模型实际大小
    uint32_t order;                             //模型order
    unordered_map<string, word_id_t> word2id;   //单词到word_id的映射
    unordered_set<GramNode *, std::hash<GramNode *>, GramNodeEqual> node_set; //所有gram的set
    GramNode *grams_per_order[MAX_ORDER];       //每个order的gram数组

    const_str_p *id2str;   // word_id到word的映射
    size_t total_gram_num; //载入时总共gram数
    size_t max_gram_num_in_all_order;

    GramNode *grams_buff;
    prob_t *bow_buff;
    GramNode **child_buff;
    size_t *child_num_buff;
    bool *need_to_recalc_bow;
    GramNode node_buff[MAX_ORDER]; //这个buff用来查询真正的node

    size_t low_gram_num;

    size_t bos_gram_id;

    //下面这些用于解析arpa文件
    static const string ngram_prefix;
    static const string data_prefix;
    static const string ngram_block_prefix;
    static const string ngram_block_suffix;
    static const string end_prefix;
    static const string sep;
    static const string escape;

    inline GramNode* gram(uint32_t cur_order, size_t gram_id,
                          word_id_t cur_word_id, GramNode *prefix, prob_t prob, prob_t bow) {
        GramNode &gram = grams_buff[gram_id];
        gram.prefix = prefix;
        gram.word_id = cur_word_id;
        gram.gram_id = gram_id;
        gram.prob = prob;
        if (low_gram_num > gram_id) {
            bow_buff[gram_id] = bow;
        }
        if (prefix) {
            gram.sib = child_buff[prefix->gram_id];
            child_buff[prefix->gram_id] = &gram;
            child_num_buff[prefix->gram_id]++;
        }
        return &gram;
    }

    template<class T> T& puts_gram(T &os, GramNode *n) const {
        GramNode *nodes[MAX_ORDER];
        uint32_t i = 0;
        while (n) {
            nodes[i++] = n;
            //os << (*id2str[n->word_id]) << " ";
            n = n->prefix;
        }
        for (int32_t j = i; j > 1; j--) {
            os << (*id2str[nodes[j - 1]->word_id]) << " ";
        }
        os << (*id2str[nodes[0]->word_id]);
        return os;
    }

    template<class T> bool load(T &in);
    bool load(const char *filename);
    double calc_distance(uint32_t lvl, GramNode *n);
    void recalc_bow();
    void recalc_bow_for_node(GramNode &node, uint32_t lvl);
    void prune_level(uint32_t lvl, size_t cut_num);
    //prob_t get_pr(GramNode *nodes, uint32_t order);
    bool computeBOW(GramNode **context, uint32_t lvl, double &numerator, double &denominator);
    prob_t wordProbBO(GramNode **context, uint32_t context_len);
public:
    LanguageModel(const char *filename)
        : order(0), id2str(NULL), total_gram_num(0), max_gram_num_in_all_order(0),
          grams_buff(NULL), bow_buff(NULL), child_buff(NULL),
          child_num_buff(NULL), need_to_recalc_bow(NULL), low_gram_num(0), bos_gram_id(-1) {
        memset(grams_per_order, 0, sizeof(grams_per_order));
        memset(ngram_num_per_order, 0, sizeof(ngram_num_per_order));
        memset(used_ngram_num_per_order, 0, sizeof(used_ngram_num_per_order));
        memset(ngram_num_order_less_than, 0, sizeof(ngram_num_order_less_than));

        node_buff[0].prefix = NULL;
        for (uint32_t i = 1; i < MAX_ORDER; i++) {
            node_buff[i].prefix = &node_buff[i - 1]; //用来在unordered_set中查询
            node_buff[i].gram_id = -1;
            node_buff[i].word_id = -1;
        }
        this->load(filename);
    }
    ~LanguageModel() {
        if (id2str != NULL) {
            delete[] id2str;
        }
        if (grams_buff) {
            delete[] grams_buff;
        }
        if (bow_buff) {
            delete[] bow_buff;
        }
        if (child_buff) {
            delete[] child_buff;
        }
        if (child_num_buff) {
            delete[] child_num_buff;
        }
        if (need_to_recalc_bow) {
            delete[] need_to_recalc_bow;
        }
    }
    uint32_t get_order() { return order; }
    bool save(const char *filename);
    template<class T> friend T&operator<<(T &os, const LanguageModel &lm);
    void prune(bool cut, uint32_t nlvl, size_t *ncut);
};
/*
prob_t LanguageModel::get_pr(GramNode *nodes, uint32_t order) {
  prob_t prob = 0;
  auto node_ = node_set.find(&nodes[order - 1]);
  if (node_ != node_set.end()) {
    prob = (*node_)->prob; // suffix prob 可以直接获取
  } else {
    //计算suffix prob
    auto prefix_node_ = node_set.find(&nodes[order - 2]);
    assert(prefix_node_ != node_set.end());
    nodes[1].prefix = NULL;
    prob = get_pr(nodes + 1, order - 1, node_set) +
           bow_buff[(*prefix_node_)->gram_id];
    nodes[1].prefix = &nodes[0];
  }
  return prob;
}*/
template<class T> bool LanguageModel::load(T &in) {
    cerr << "[LOADING] start loading..." << endl;
    string str;

    while (getline(in, str) && !boost::algorithm::starts_with(str, data_prefix)) {}
    while (getline(in, str) && boost::algorithm::starts_with(str, ngram_prefix)) {
        uint32_t cur_order = atoi(str.c_str() + ngram_prefix.length());
        order = max(this->order, cur_order);
        size_t offset = str.find_first_of("=");
        assert(offset != string::npos);
        ngram_num_per_order[cur_order - 1] = atol(str.c_str() + offset + 1);
        total_gram_num += ngram_num_per_order[cur_order - 1];
    }
    id2str = new const_str_p[ngram_num_per_order[0]]();
    grams_buff = new GramNode[total_gram_num];
    low_gram_num = total_gram_num - ngram_num_per_order[order - 1];
    child_buff = new GramNode *[low_gram_num]();
    child_num_buff = new size_t[low_gram_num]();
    bow_buff = new prob_t[low_gram_num]; //最高order的gram没有bow
    need_to_recalc_bow = new bool[low_gram_num]();
    size_t offset = 0;
    for (uint32_t i = 0; i < order; i++) {
        ngram_num_order_less_than[i] = offset;
        max_gram_num_in_all_order =
            max(max_gram_num_in_all_order, ngram_num_per_order[i]);
        grams_per_order[i] = grams_buff + offset; //初始化了么？
        offset += ngram_num_per_order[i];
    }
    assert(order > 1 && "[ERR]language model order is 1."); //不适用于一元模型
    node_set.reserve(total_gram_num);
    word2id.reserve(ngram_num_per_order[0]);

    uint32_t cur_order = 0;
    uint32_t gram_id = 0;
    uint32_t cur_word_id = 0;
    while (getline(in, str)) {
        if (boost::algorithm::starts_with(str, ngram_block_prefix)) {
            if (boost::algorithm::starts_with(str, end_prefix)) {
                break;
            }
            assert(boost::algorithm::ends_with(str, ngram_block_suffix));
            cur_order = atoi(str.c_str() + 1);
            continue;
        } else if (str.length() == 0) {
            continue;
        }
        prob_t prob = atof(str.c_str());
        size_t token_start = 0;
        size_t token_end = str.find_first_of("\t ");
        for (uint32_t i = 0; i < cur_order; i++) {
            token_start = str.find_first_not_of("\t ", token_end);
            token_end = str.find_first_of("\t ", token_start);
            string word = str.substr(token_start, token_end - token_start);
            auto got = word2id.find(word);
            if (got != word2id.end()) {
                node_buff[i].word_id = got->second;
            } else {
                node_buff[i].word_id = cur_word_id;
                auto inserted = word2id.insert(make_pair(word, cur_word_id));
                id2str[cur_word_id] = &inserted.first->first;
                cur_word_id++;
            }
            //cerr << node_buff[i].word_id << endl;
        }
        prob_t bow = 0;
        if (token_end != string::npos) {
            bow = atof(str.c_str() + token_end);
        }
        //puts_gram(cerr, &node_buff[cur_order-1]);
        //cerr << endl;
        GramNode *prefix = NULL;

        if (cur_order >= 2) { //只有在二元及更高元，才会考虑prefix和suffix
            auto prefix_ = node_set.find(&node_buff[cur_order - 2]);
            if (prefix_ == node_set.end()) {
                cerr << "[ERR] gram prefix should also be in language model. but cannot find: ";
                puts_gram(cerr, &node_buff[cur_order - 2]);
                cerr << endl;
                assert(prefix_ != node_set.end());
            }
            prefix = (*prefix_);
        }
        GramNode *gram =
            this->gram(cur_order, gram_id++, node_buff[cur_order - 1].word_id,
                       prefix, prob, bow);
        node_set.insert(gram);
        used_ngram_num_per_order[cur_order - 1]++;
        //cerr << str << "\t";
        //puts_gram(cerr, gram);
        //cerr << endl;

        //srilm的做法是
        //如果存在<s>那么将<s>暂时概率设为</s>的概率.否则计算P(H)会不正确。
        //在最后保存的时候再设为0

        auto bos_ = word2id.find(bos);
        if (bos_ != word2id.end()) {
            //该语言模型中包含bos
            node_buff[0].word_id = bos_->second;
            auto bos_gram_ = node_set.find(&node_buff[0]);
            assert(bos_gram_ != node_set.end());

            bos_gram_id = (*bos_gram_)->gram_id;
            auto eos_ = word2id.find(eos);
            assert(eos_ != word2id.end());
            node_buff[0].word_id = eos_->second;
            auto eos_gram_ = node_set.find(&node_buff[0]);
            assert(eos_gram_ != node_set.end());
            (*bos_gram_)->prob = (*eos_gram_)->prob;
        }
    }

    assert(cur_word_id <= used_ngram_num_per_order[0]&& "all words must be in unigram");
    for (uint32_t o = 0; o < order; o++) {
        cerr << "[LOADING] ngram " << (o + 1) << " = " << used_ngram_num_per_order[o] << endl;
        assert(ngram_num_per_order[o] == used_ngram_num_per_order[o] &&
               "[ERR] ngame num not match");
    }
    cerr << "[LOADING] finished." << endl;
    return true;
}

bool LanguageModel::load(const char *filename) {
    ifstream file(filename, ios_base::in | ios_base::binary);
    boost::iostreams::filtering_istream in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(file);
    load(in);
    file.close();
    return true;
}

bool LanguageModel::save(const char *filename) {
    cerr << "[SAVING] save pruned language model ..." << endl;
    //首先需要将bos的概率恢复0.
    grams_buff[bos_gram_id].prob = min_log_prob;
    ofstream file(filename, ios_base::out | ios_base::binary);
    boost::iostreams::filtering_ostream out;
    out.push(boost::iostreams::gzip_compressor());
    out.push(file);
    out << (*this);
    file.close();
    return true;
}

const string LanguageModel::ngram_prefix = "ngram";
const string LanguageModel::data_prefix = "\\data\\";
const string LanguageModel::ngram_block_prefix = "\\";
const string LanguageModel::end_prefix = "\\end\\";
const string LanguageModel::ngram_block_suffix = "-grams:";
const string LanguageModel::sep = "\t";
const string LanguageModel::escape = "\\";
template<class T>
T& operator<<(T &os, const LanguageModel &lm) {
    os << endl << LanguageModel::data_prefix << endl;
    for (uint32_t i = 0; i < lm.order; i++) {
        os << LanguageModel::ngram_prefix << " " << i << "=" << lm.used_ngram_num_per_order[i] << endl;
    }
    os << endl;
    for (uint32_t lvl = 0; lvl < lm.order; lvl++) {
        os << LanguageModel::escape << (lvl + 1) << LanguageModel::ngram_block_suffix << endl;
        size_t gram_num = lm.ngram_num_per_order[lvl];
        GramNode *start_node = lm.grams_per_order[lvl];
        GramNode *end_node = start_node + gram_num;
        for (GramNode *cur_node = start_node; cur_node != end_node; cur_node++) {
            if (cur_node->prob > 2.0) {
                continue; //这个gram已经被删除了
            }
            os << cur_node->prob << LanguageModel::sep;
            lm.puts_gram(os, cur_node);
            if (lm.bow_buff[cur_node->gram_id] != 0.0) {
                os << LanguageModel::sep << lm.bow_buff[cur_node->gram_id];
            }
            os << endl;
        }
        os << endl;
    }
    os << LanguageModel::end_prefix << endl;
    return os;
}

//从最高order的ngram开始进行prune
void LanguageModel::prune(bool cut, uint32_t nlvl, size_t *ncut) {
    for (int32_t lvl = nlvl - 1; lvl >= 0; lvl--) {
        size_t cut_size = ncut[lvl];
        if (!cut) {
            if (used_ngram_num_per_order[lvl] > ncut[lvl]) {
                cut_size = used_ngram_num_per_order[lvl] - ncut[lvl];
            } else {
                cut_size = 0;
            }
        }
        if (cut_size == 0) {
            cerr << "[PRUNING][ORDER=" << (lvl + 1) << "] no need to cut." << endl;
            continue;
        }
        prune_level(lvl, cut_size);
    }
    recalc_bow();
}

void LanguageModel::prune_level(uint32_t lvl, size_t cut_num) {
    assert(lvl >= 1 && "cannot prune unigram");
    cerr <<  "[PRUNING][ORDER=" << (lvl + 1) << "] cut_num = " << cut_num << endl;
    cerr << "[PRUNING][ORDER=" << (lvl + 1) << "] calculate entropy distance..." << endl;
    size_t num_in_level = used_ngram_num_per_order[lvl];
    if (cut_num > num_in_level) {
        cut_num = num_in_level;
    }
    size_t total_num_in_level = ngram_num_per_order[lvl];
    GramInfo *node_info_buff = new GramInfo[num_in_level];
    size_t buff_idx = 0; //当前有多少可供裁剪的gram
    for (size_t i = 0; i < total_num_in_level; i++) {
        GramNode &n = grams_per_order[lvl][i];
        if (n.prob > 2.0) { //这个概率是invalid的，代表这个gram已经被删除
            continue;
        }
        GramInfo &info = node_info_buff[buff_idx++];
        if (n.gram_id < low_gram_num && child_num_buff[n.gram_id] != 0) {
            //有更高阶的gram依赖这个gram，因此不能删除
            continue;
        }
        info.node = &n;
        info.dis = calc_distance(lvl, &n);
    }
    cerr << "[PRUNING][ORDER=" << (lvl + 1) << "] sorting..." << endl;
    make_heap(node_info_buff, node_info_buff + buff_idx);
    sort_heap(node_info_buff, node_info_buff + buff_idx);
    size_t cutted = 0;
    for (size_t i = 0; i < buff_idx && cutted < cut_num; i++) {
        //将这些熵的增量小的都可以删除
        GramNode *node = node_info_buff[i].node;
        node->prob = 3.0; //用不合法的prob来标记当前的node被删除
        need_to_recalc_bow[node->prefix->gram_id] = true;
        used_ngram_num_per_order[lvl]--;
        child_num_buff[node->prefix->gram_id]--;
    }
    delete[] node_info_buff;
    cerr << "[PRUNING][ORDER=" << (lvl + 1) << "] finished." << endl;
}

//gram[context_len]是P(w|h)中的w
prob_t LanguageModel::wordProbBO(GramNode **gram, uint32_t context_len) {
    //返回P(w|h')的log值
    //srilm的计算方法和sunpinyin的不同。
    //srilm不需要递归计算
    //srilm首先计算最长的P(w|h')。然后加上bow值
    cerr << "wordProbBO: ";
    puts_gram(cerr, gram[context_len]);
    cerr << endl;

    auto node_found_ = node_set.find(gram[context_len]);
    if (node_found_ != node_set.end()) {
        return (*node_found_)->prob;
    }
    assert(context_len != 0 && "Unigram must in Language model");
    //查找最长的在语言模型中的h'w的suffix
    prob_t prob = 3.0; //用大于0的数代表没有找到。
    uint32_t i = 0; //i+1是去掉的head的数目。也是需要补的bow的数目
    for (; i < context_len; i++) {
        gram[i + 1]->prefix = NULL;
        auto prob_node_found_ = node_set.find(gram[context_len]);
        gram[i + 1]->prefix = gram[i];
        if (prob_node_found_ != node_set.end()) {
            prob = (*prob_node_found_)->prob;
            break;
        }
    }
    if (prob >= 2.0) {
        cerr << "prob must be found: ";
        puts_gram(cerr, gram[context_len]);
        cerr << endl;
        assert(prob < 2.0 && "prob must be found");
    }
    for (uint32_t j = 0; j <= i; j++) {
        auto bow_node_found_ = node_set.find(gram[context_len-1-j]);
        assert(bow_node_found_ != node_set.end() && "bow must be found");
        GramNode *bow_node = (*bow_node_found_);
        prob += bow_buff[bow_node->gram_id];
    }
    return prob;
}

//副作用会改变context[lvl]
bool LanguageModel::computeBOW(GramNode **context, uint32_t lvl, double &numerator, double &denominator) {
    numerator = 1.0;
    denominator = 1.0;
    assert(lvl >= 1); //不能对unigram调用
    cerr << "computeBOW: ";
    puts_gram(cerr, context[lvl - 1]);
    cerr << endl;
    GramNode *child = child_buff[context[lvl - 1]->gram_id]; //和context[0]是兄弟的节点

    context[1]->prefix = NULL;
    while (child) {
        numerator -= exp10(child->prob);
        //srilm在此处判断了lvl的值。但是似乎在prune的调用中不需要
        //计算P(w|h'), h' 是suffix(h)
        //如果当前的lvl是1,那么h是uningram,suffix(h)为空。
        context[lvl] = child;
        denominator -= exp10(wordProbBO(context + 1, lvl - 1));
        child = child->sib;
    }
    context[1]->prefix = context[0];
    assert(numerator > -0.01 && denominator > -0.01);
    if (numerator < 0.00001 || denominator < 0.00001) { //出现了精度问题
        cerr << "precision problem on ";
        puts_gram(cerr, context[lvl - 1]);
        cerr << "\t";
        if (numerator < 0.00001) {
            cerr << "{1.0 - sigma p(w|h)} ==> 0.00001" << endl;
            numerator = 0.00001;
        }
        if (denominator < 0.00001) {
            cerr << "{1.0 - sigma p(w|h')} ==> 0.00001" << endl;
            denominator = 0.00001;
        }
    }
    return true;
}

double LanguageModel::calc_distance(uint32_t lvl, GramNode *n) {
    GramNode *prefix = n->prefix;
    prob_t PH = 0.0;
    while (prefix != NULL) {
        PH += prefix->prob;
        prefix = prefix->prefix;
    } //感觉此处sunpinyin错了？
    cerr << "calcing distance: ";
    puts_gram(cerr, n);
    cerr << endl;
    prefix = n->prefix;
    prob_t log10_BOW = bow_buff[prefix->gram_id];

    prob_t BOW = exp10(log10_BOW); //bow(h)
    GramNode *context[MAX_ORDER];

    //0       1         lvl-1  lvl
    //w_0  .........  w_{n-1} w_n
    for (int32_t i = lvl; i >= 0; i--) {
        context[i] = n;
        n = n->prefix;
    }
    n = context[lvl];

    //double numerator, denominator;
    prob_t PA, PB;
    computeBOW(context, lvl, PA, PB); //srilm中如果算出来不对了，那么不删除，有可能因为精度关系概率<0,sunpinyin设了一个较小的值
    PH = exp10(PH); // PH是当前gram的history出现的概率。P(h)
    if (!(PH <= 1.0 && PH > 0)) {
        cerr << "[ERR] history prob not in (0, 1.0], gram_id = " << prefix->gram_id << ", P(";
        puts_gram(cerr, prefix);
        cerr << ") = " << PH << endl;
        while (prefix != NULL) {
            cerr << "gram_id = " << prefix->gram_id << "\t, conditional prob = " << prefix->prob << endl;
            prefix = prefix->prefix;
        }
        assert(PH <= 1.0 && PH > 0);
    }

    prob_t log10_PHW = n->prob;
    prob_t PHW = exp10(log10_PHW); // P(w|h)
                                   // h'是h的suffix.
    context[lvl] = n;
    context[1]->prefix = NULL;
    prob_t log10_PH_W = wordProbBO(context + 1, lvl - 1); // P(w|h')
    context[1]->prefix = context[0];
    //在sunpinyin中, log10_PH_W 不是直接获得，而是需要计算。
    // language model中不一定有这个gram

    prob_t PH_W = exp10(log10_PH_W); // P(head(h)|tail(h))
    assert(PHW < 1.0 && PHW > 0.0);
    assert(PH_W < 1.0 && PH_W > 0.0);

    prob_t _BOW = (PA + PHW) / (PB + PH_W);
    assert(BOW > 0.0);
    assert(_BOW > 0.0);
    assert(PA + PHW < 1.01);  // %1 error rate
    assert(PB + PH_W < 1.01); // %1 error rate
    prob_t log10__BOW = log(_BOW);
    return -(PH * (PHW * (log10_PH_W + log10__BOW - log10_PHW) +
                   PA * (log10__BOW - log10_BOW)));
}
void LanguageModel::recalc_bow_for_node(GramNode &node, uint32_t lvl) {
    prob_t sum_next = 0;
    prob_t sum = 0;
    if (child_num_buff[node.gram_id] == 0) {
        bow_buff[node.gram_id] = 0; //当前bow==1，log bow == 0
    }
    GramNode *context[MAX_ORDER];
    GramNode *n = &node;
    for (int32_t i = lvl; i >= 0; i--) {
        context[i] = n;
        n = n->prefix;
    }

    prob_t PA, PB;
    computeBOW(context, lvl, PA, PB);
    if (lvl == 0) {
        if (PA < Prob_Epsilon) {
            /*
             * Avoid spurious non-zero unigram probabilities
             */
            PA = 0.0;
        }
    } else if (PA < Prob_Epsilon && PB < Prob_Epsilon) {
        bow_buff[node.gram_id] = LogP_One;
    } else {
        bow_buff[node.gram_id] = log10(PA) - log10(PB);
    }

    assert(sum_next >= 0.0 && sum_next < 1.0);
    assert(sum >= 0.0 && sum < 1.0);
    prob_t new_bow = log10((1.0 - sum_next) / (1.0 - sum));
    bow_buff[node.gram_id] = new_bow;
}

void LanguageModel::recalc_bow() {
    uint32_t lvl = 1;
    cerr << "[RECALC_BOW] start to recalculate bow..." << endl;
    for (size_t i = 0; i < low_gram_num; i++) {
        //重新计算bow值
        if (grams_buff[i].prob < 2.0 && need_to_recalc_bow[i]) { //没有被删除，且需要计算bow
            recalc_bow_for_node(grams_buff[i], lvl - 1);
        }
        if (i > ngram_num_order_less_than[lvl]) {
            cerr << "[RECALC_BOW] order " << lvl << " grams' bow are done" << endl;
            lvl++;
        }
    }
    cerr <<  "[RECALC_BOW] order " << lvl << " grams' bow are done" << endl;
}

static const char USAGE[] =
    R"(Prune language model.
    This program uses entropy - based method to prune the size of back-off\
    language model 'src_model' to a specific size and write to 'dst_model'.
Note that we do not ensure that during pruning process,  exactly the\
the given number of items are cut or reserved, because some items may\
contains high level children, so could not be cut.
param <count> format example: 1=100 2=3000 3=4000
Usage:
lmprune entropy (reserve|cut) <src_model> <dst_model> <count>...
lmprune (-h | --help)
lmprune --version

Options:
-h --help     Show this screen.
--version     Show version.
)";

int main(int argc, char *argv[]) {
    map<string, docopt::value> args =
        docopt::docopt(USAGE, { argv + 1, argv + argc },
                       true, // show help if requested
                       "lmprune 1.0");
    auto entropy_method_ = args.find("entropy");
    assert(entropy_method_ != args.end() && entropy_method_->second.isBool());
    assert(entropy_method_->second.asBool()&&
           "currently only support entropy method");
    auto is_reserve_ = args.find("reserve");
    auto is_cut_ = args.find("cut");
    assert(is_reserve_ != args.end()&& is_reserve_->second.isBool());
    assert(is_cut_ != args.end()&& is_cut_->second.isBool());
    bool is_reserve = is_reserve_->second.asBool();
    bool is_cut = is_cut_->second.asBool();
    assert(is_reserve != is_cut);
    auto src_model_ = args.find("<src_model>");
    auto dst_model_ = args.find("<dst_model>");
    assert(src_model_ != args.end()&& src_model_->second.isString());
    assert(dst_model_ != args.end()&& dst_model_->second.isString());
    const string &src_model = src_model_->second.asString();
    const string &dst_model = dst_model_->second.asString();
    auto ncut_str_ = args.find("<count>");
    assert(ncut_str_ != args.end()&& ncut_str_->second.isStringList());
    const vector<string> &ncut_str = ncut_str_->second.asStringList();
    LanguageModel lm(src_model.c_str());
    uint32_t order = lm.get_order();
    size_t ncut[order];
    memset(ncut, -1, sizeof(ncut));
    uint32_t max_order = 0;
    for (auto it = ncut_str.begin(); it != ncut_str.end(); ++it){
        size_t eq_offset = it->find_first_of("=");
        if (eq_offset == string::npos){
            cerr << "<count> format is illegal. it should be order=num" << endl;
            exit(1);
        }
        uint32_t cur_order = atoi(it->c_str());
        max_order = max(max_order, cur_order);
        assert(cur_order <= order && "<count> param contains gram order bigger than language model's order");
        assert(cur_order != 1 && "cannot prune unigram");
        size_t num = atol(it->c_str()+ eq_offset + 1);
        ncut[cur_order-1] = num;
    }
    lm.prune(is_cut, max_order, ncut);
    lm.save(dst_model.c_str());
}


