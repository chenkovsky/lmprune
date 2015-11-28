#ifndef __GRAM_HPP__
#define __GRAM_HPP__ 1
#include <cstdlib>
#define MAX_ORDER 10
typedef uint32_t word_id_t;
typedef double prob_t;

struct GramNode {
    word_id_t word_id;
    size_t gram_id;
    struct GramNode *prefix;
    bool operator==(const GramNode &r) const;
    uint32_t words(word_id_t* ret_arr) const;//将自己的所有word id写入数组,逆序,返回长度
    GramNode(): word_id(-1), gram_id(-1),prefix(NULL){
    }
};

#endif

