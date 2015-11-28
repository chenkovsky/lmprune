#include "docopt.cpp/docopt.h"
#include "lm.hpp"
bool is_debug = false;
static const char USAGE[] =
R"(Prune language model.
This program uses entropy - based method to prune the size of back-off\
language model 'src_model' to a specific size and write to 'dst_model'.
Note that we do not ensure that during pruning process,  exactly the\
the given number of items are cut or reserved, because some items may\
contains high level children, so could not be cut.
param <count> format example: 1=100 2=3000 3=4000
Usage:
    lmprune entropy (reserve|cut) [--debug] [--important=important_ngram] <src_model> <dst_model> <count>...
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
    auto is_debug_ = args.find("--debug");
    assert(is_debug_ != args.end() && is_debug_->second.isBool());
    is_debug = is_debug_->second.asBool();

    auto src_model_ = args.find("<src_model>");
    auto dst_model_ = args.find("<dst_model>");
    auto important_ = args.find("--important");
    assert(src_model_ != args.end() && src_model_->second.isString());
    assert(dst_model_ != args.end()&& dst_model_->second.isString());
    const string &src_model = src_model_->second.asString();
    const string &dst_model = dst_model_->second.asString();
    auto ncut_str_ = args.find("<count>");
    assert(ncut_str_ != args.end()&& ncut_str_->second.isStringList());
    const vector<string> &ncut_str = ncut_str_->second.asStringList();
    LanguageModel* lm = LanguageModel::load(src_model.c_str());
    if (important_ != args.end() && important_->second.isString()) {
        const string &important_file = important_->second.asString();
        lm->load_important_gram(important_file.c_str());
    }
    uint32_t order = lm->get_order();
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
    lm->prune(is_cut, max_order, ncut);
    lm->save(dst_model.c_str());
    delete lm;
}



