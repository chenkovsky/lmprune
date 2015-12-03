#include "gram.hpp"
bool GramNode::operator==(const GramNode &r) const{
    if (gram_id != (size_t)-1 && r.gram_id != (size_t)-1) {
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

uint32_t GramNode::words(word_id_t* ret_arr) const{
    uint32_t len = 0;
    const GramNode* n = this;
    while (n) {
        ret_arr[len++] = n->word_id;
        n = n->prefix;
    }
    return len;
}
