#include<iomanip>
#include <sstream>      // std::istringstream
#include "lm.hpp"
template <class T> LanguageModel* LanguageModel::load(T &in) {
    LanguageModel *lm = NULL;
    cerr << "[LOADING] start loading..." << endl;
    string str;
    size_t ngram_num_per_order[MAX_ORDER];
    word_id_t word_ids[MAX_ORDER];
    uint32_t order = 0;
    memset(ngram_num_per_order, 0, sizeof(ngram_num_per_order));
    while (getline(in, str) && !boost::algorithm::starts_with(str, data_prefix)) {}
    while (getline(in, str) && boost::algorithm::starts_with(str, ngram_prefix)) {
        uint32_t cur_order = atoi(str.c_str() + ngram_prefix.length());
        order = max(order, cur_order);
        size_t offset = str.find_first_of("=");
        assert(offset != string::npos);
        ngram_num_per_order[cur_order - 1] = atol(str.c_str() + offset + 1);
    }
    lm = new LanguageModel(ngram_num_per_order, order);

    uint32_t cur_order = 0;
    while (getline(in, str)) {
        if (boost::algorithm::starts_with(str, ngram_block_prefix)) {
            if (boost::algorithm::starts_with(str, end_prefix)) {
                break;
            }
            assert(boost::algorithm::ends_with(str, ngram_block_suffix));
            cur_order = atoi(str.c_str() + 1);
            cerr << "[LOADING] order = " << cur_order << endl;
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
            word_ids[i] = lm->word2Idx_append_if_not_exist(word);
        }
        prob_t bow = 0;
        if (token_end != string::npos) {
            bow = atof(str.c_str() + token_end);
        }
        lm->add_gram(word_ids, cur_order, prob, bow);
    }
    lm->load_finish();
    cerr << "[LOADING] load finish." << endl;
    return lm;
}

LanguageModel* LanguageModel::load(const char *filename) {
  ifstream file(filename, ios_base::in | ios_base::binary);
  boost::iostreams::filtering_istream in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(file);
  LanguageModel* ret= load(in);
  file.close();
  return ret;
}

void LanguageModel::save(const char *filename) {
    cerr << "[SAVING] save pruned language model ..." << endl;
    //首先需要将bos的概率恢复0.
    prob_buff[bos_gram_id] = min_log_prob;
    ofstream file(filename, ios_base::out | ios_base::binary);
    boost::iostreams::filtering_ostream out;
    out.push(boost::iostreams::gzip_compressor());
    out.push(file);
    out << setprecision(7);
    out << (*this);
    boost::iostreams::close(out);
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
        os << LanguageModel::ngram_prefix << " " << (i+1) << "=" << lm.used_ngram_num_per_order[i] << endl;
    }
    os << endl;
    size_t offset = 0;
    for (uint32_t lvl = 0; lvl < lm.order; lvl++) {
        cerr << "[SAVING] order = " << (lvl+1) << endl;
        os << LanguageModel::escape << (lvl + 1) << LanguageModel::ngram_block_suffix << endl;
        size_t gram_num = lm.ngram_num_per_order[lvl];
        GramNode *start_node = &lm.grams_buff[offset];
        GramNode *end_node = start_node + gram_num;
        for (GramNode *cur_node = start_node; cur_node != end_node; cur_node++) {
            if (lm.prob_buff[cur_node->gram_id] > 2.0) {
                continue; //这个gram已经被删除了
            }
            os << lm.prob_buff[cur_node->gram_id] << LanguageModel::sep;
            lm.puts_gram(os, cur_node);
            //lm.puts_gram(cerr, cur_node);
            //cerr << endl;
            if (lm.bow_buff[cur_node->gram_id] != 0.0) {
                os << LanguageModel::sep << lm.bow_buff[cur_node->gram_id];
            }
            os << endl;
        }
        os << endl;
        offset += lm.ngram_num_per_order[lvl];
    }
    os << LanguageModel::end_prefix << endl;
    return os;
}

template<class T> void LanguageModel::load_important_gram(T &in){
    string str;
    while (getline(in, str)) {
        istringstream iss(str);
        vector<string> tokens{istream_iterator<string>{iss},
                      istream_iterator<string>{}};
        const GramNode* gram_ = gram(tokens);
        if (gram_ != NULL) {
            mask_for_grams[gram_->gram_id] |= DONT_PRUNE;
        }
    }
}

void LanguageModel::load_important_gram(const char* filename){
  ifstream file(filename, ios_base::in | ios_base::binary);
  boost::iostreams::filtering_istream in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(file);
  load_important_gram(in);
  file.close();
}
