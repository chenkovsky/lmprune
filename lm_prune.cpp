#include "lm.hpp"
extern bool is_debug;
struct GramInfo {
    GramNode *node;
    double dis; // distance
    bool operator<(const GramInfo &r) const { return dis < r.dis; }
    bool operator==(const GramInfo &r) const { return dis == r.dis; }
};

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
        if (mask_for_grams[n.gram_id] & DONT_PRUNE) {
            continue;
        }
        if (prob_buff[n.gram_id] > 2.0) { //这个概率是invalid的，代表这个gram已经被删除
            continue;
        }
        if (n.gram_id < low_gram_num && child_num_buff[n.gram_id] != 0) {
            //有更高阶的gram依赖这个gram，因此不能删除
            continue;
        }
        GramInfo &info = node_info_buff[buff_idx++];
        info.node = &n;
        info.dis = calc_distance(lvl, &n);
    }
    cerr << "[PRUNING][ORDER=" << (lvl + 1) << "] gram_num_can_be_removed = "<< buff_idx <<", sorting..." << endl;
    make_heap(node_info_buff, node_info_buff + buff_idx);
    sort_heap(node_info_buff, node_info_buff + buff_idx);
    size_t cutted = 0;
    for (size_t i = 0; i < buff_idx && cutted < cut_num; i++) {
        //将这些熵的增量小的都可以删除
        cutted++;
        GramNode *node = node_info_buff[i].node;
        prob_buff[node->gram_id] = 3.0; //用不合法的prob来标记当前的node被删除
        mask_for_grams[node->prefix->gram_id] |= PRUNED;
        used_ngram_num_per_order[lvl]--;
        child_num_buff[node->prefix->gram_id]--;
    }
    delete[] node_info_buff;
    cerr << "[PRUNING][ORDER=" << (lvl + 1) << "] finished." << endl;
}


//gram[context_len]是P(w|h)中的w
prob_t LanguageModel::wordProbBO(word_id_t *word_ids, uint32_t context_len) {
    //返回P(w|h')的log值
    //srilm的计算方法和sunpinyin的不同。
    //srilm不需要递归计算
    //srilm首先计算最长的P(w|h')。然后加上bow值
    const GramNode *n = gram(word_ids, context_len + 1);
    if (n && prob_buff[n->gram_id] < 2.0) {
        return prob_buff[n->gram_id];
    }
    assert(context_len != 0 && "Unigram must in Language model");
    //查找最长的在语言模型中的h'w的suffix
    prob_t prob = 3.0; //用大于0的数代表没有找到。
    uint32_t i = 0; //i+1是去掉的head的数目。也是需要补的bow的数目
    for (; i < context_len; i++) {
        n = gram(word_ids + i + 1, context_len - i);
        if (n) {
            prob = prob_buff[n->gram_id];
            if (prob < 2.0) {
                break;
            }
        }
    }
    assert(prob < 2.0 && "prob must be found");

    for (uint32_t j = 0; j <= i; j++) {
        n = gram(word_ids + j, context_len - j);
        assert(n != NULL && "bow must be found");
        prob += bow_buff[n->gram_id];
    }
    return prob;
}

//副作用会改变context[lvl]
bool LanguageModel::computeBOW(size_t gram_id, word_id_t *context, uint32_t lvl, double &numerator, double &denominator) {
    numerator = 1.0;
    denominator = 1.0;
    assert(lvl >= 1); //不能对unigram调用
    GramNode *child = child_buff[gram_id]; //和context[0]是兄弟的节点

    while (child) {
        if (prob_buff[child->gram_id] > 2.0) {
            child = sib_buff[child->gram_id];
            continue;
        }
        numerator -= exp10(prob_buff[child->gram_id]);
        //srilm在此处判断了lvl的值。但是似乎在prune的调用中不需要
        //计算P(w|h'), h' 是suffix(h)
        //如果当前的lvl是1,那么h是uningram,suffix(h)为空。
        context[lvl] = child->word_id;
        denominator -= exp10(wordProbBO(context + 1, lvl - 1));
        child = sib_buff[child->gram_id];
    }

    assert(numerator > -0.01 && denominator > -0.01);
    if (numerator < 0.00001 || denominator < 0.00001) { //出现了精度问题
        cerr << "precision problem on ";
        puts_gram(cerr, &grams_buff[gram_id]);
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
    prob_t log10_PH = 0.0;
    while (prefix != NULL) {
        log10_PH += prob_buff[prefix->gram_id];
        prefix = prefix->prefix;
    } //感觉此处sunpinyin错了？
    prefix = n->prefix;
    prob_t log10_BOW = bow_buff[prefix->gram_id];

    prob_t BOW = exp10(log10_BOW); //bow(h)
    word_id_t context[MAX_ORDER];

    //0       1         lvl-1  lvl
    //w_0  .........  w_{n-1} w_n
    GramNode *tmp = n;
    for (int32_t i = lvl; i >= 0; i--) {
        context[i] = tmp->word_id;
        tmp = tmp->prefix;
    }
    //double numerator, denominator;
    prob_t PA, PB;
    computeBOW(n->prefix->gram_id, context, lvl, PA, PB); //srilm中如果算出来不对了，那么不删除，有可能因为精度关系概率<0,sunpinyin设了一个较小的值
    prob_t PH = exp10(log10_PH); // PH是当前gram的history出现的概率。P(h)
    if (!(PH <= 1.0 && PH > 0)) {
        cerr << "[ERR] history prob not in (0, 1.0], gram_id = " << prefix->gram_id << ", P(";
        puts_gram(cerr, prefix);
        cerr << ") = " << PH << endl;
        while (prefix != NULL) {
            cerr << "gram_id = " << prefix->gram_id << "\t, conditional prob = " << prob_buff[prefix->gram_id] << endl;
            prefix = prefix->prefix;
        }
        assert(PH <= 1.0 && PH > 0);
    }

    prob_t log10_PHW = prob_buff[n->gram_id];
    prob_t PHW = exp10(log10_PHW); // P(w|h)
                                   // h'是h的suffix.
    context[lvl] = n->word_id;
    prob_t log10_PH_W = wordProbBO(context + 1, lvl - 1); // P(w|h')
                                                          //在sunpinyin中, log10_PH_W 不是直接获得，而是需要计算。
                                                          // language model中不一定有这个gram

    prob_t PH_W = exp10(log10_PH_W); // P(head(h)|tail(h))
    assert(PHW < 1.0 && PHW > 0.0);
    assert(PH_W < 1.0 && PH_W > 0.0);

    
    assert(BOW > 0.0);
    assert(PA + PHW < 1.01);  // %1 error rate
    assert(PB + PH_W < 1.01); // %1 error rate
    prob_t log10__BOW = log10(PA + PHW) - log10(PB + PH_W);
    prob_t _BOW = exp10(log10__BOW);
    assert(_BOW > 0.0);
    double deltaEntropy = -(PH * (PHW * (log10_PH_W + log10__BOW - log10_PHW) +
                   PA * (log10__BOW - log10_BOW)));
    if(is_debug){
        cerr << "GRAM ";
        puts_gram(cerr, n);
        cerr << " CONTEXTPROB " << log10_PH
        << " OLDPROB " << log10_PHW
        << " NEWPROB " << (log10_PH_W + log10__BOW)
        << " OLDBOW " << log10_BOW
        << " NEWBOW " << log10__BOW
        << " DELTA-H " << deltaEntropy
        << " DELTA-LOGP " << (log10_PH_W + log10__BOW - prob_buff[n->gram_id])
        << " PA " << PA
        << " PB " << PB
        << " PH_W " << log10_PH_W
        << endl;
    }
    return deltaEntropy;
}
void LanguageModel::recalc_bow_for_node(GramNode &node, uint32_t lvl) {
    //prob_t sum_next = 0;
    //prob_t sum = 0;
    if (child_num_buff[node.gram_id] == 0) {
        bow_buff[node.gram_id] = 0; //当前bow==1，log bow == 0
    }
    word_id_t context[MAX_ORDER];
    GramNode *tmp = &node;
    for (int32_t i = lvl; i >= 0; i--) {
        context[i] = tmp->word_id;
        tmp = tmp->prefix;
    }

    prob_t PA, PB;
    computeBOW(node.gram_id, context, lvl + 1, PA, PB);
    /*if (lvl == 0) {
        if (PA < Prob_Epsilon) {
            
            PA = 0.0;
        }
    } else */
    if (PA < Prob_Epsilon && PB < Prob_Epsilon) {
        bow_buff[node.gram_id] = LogP_One;
    } else {
        bow_buff[node.gram_id] = log10(PA) - log10(PB);
    }
    if (is_debug) {
        cerr << "[RECALC_BOW]";
        puts_gram(cerr, &node);
        cerr << " " << bow_buff[node.gram_id] << endl;
    }
    //assert(sum_next >= 0.0 && sum_next < 1.0);
    //assert(sum >= 0.0 && sum < 1.0);
    //prob_t new_bow = log10((1.0 - sum_next) / (1.0 - sum));
    //bow_buff[node.gram_id] = new_bow;
}

void LanguageModel::recalc_bow() {
    uint32_t lvl = 0;
    cerr << "[RECALC_BOW] start to recalculate bow..." << endl;
    size_t offset = 0;
    for (size_t i = 0; i < low_gram_num; i++) {
        //重新计算bow值
        if (prob_buff[i] < 2.0 && (mask_for_grams[i] & PRUNED)) { //没有被删除，且需要计算bow
            recalc_bow_for_node(grams_buff[i], lvl);
        }
        if (i - offset > ngram_num_per_order[lvl]) {
            cerr << "[RECALC_BOW] order " << (lvl+1) << " grams' bow are done" << endl;
            offset += ngram_num_per_order[lvl];
            lvl++;
        }
    }
    cerr <<  "[RECALC_BOW] order " << (lvl+1) << " grams' bow are done" << endl;
}
