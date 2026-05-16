/**
 * Adapted from PHF reference (origin/v1.1.0_dev:modules/codec/huffman/hf_bk_impl1.seq.cc)
 * Original authors: Sheng Di, Jiannan Tian (Argonne National Laboratory, Indiana University)
 * Changes:
 *   - Removed #include "timer.hh" (not used in this file's function bodies).
 */

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hf_impl.hh"

HuffmanTree* create_tree_serial(int state_num)
{
    auto ht = (HuffmanTree*)malloc(sizeof(HuffmanTree));
    memset(ht, 0, sizeof(HuffmanTree));
    ht->state_num = state_num;
    ht->all_nodes = 2 * state_num;

    ht->pool = (struct node_t*)malloc(ht->all_nodes * 2 * sizeof(struct node_t));
    ht->qqq  = (node_list*)malloc(ht->all_nodes * 2 * sizeof(node_list));
    ht->code = (uint64_t**)malloc(ht->state_num * sizeof(uint64_t*));
    ht->cout = (uint8_t*)malloc(ht->state_num * sizeof(uint8_t));

    memset(ht->pool, 0, ht->all_nodes * 2 * sizeof(struct node_t));
    memset(ht->qqq,  0, ht->all_nodes * 2 * sizeof(node_list));
    memset(ht->code, 0, ht->state_num * sizeof(uint64_t*));
    memset(ht->cout, 0, ht->state_num * sizeof(uint8_t));
    ht->qq     = ht->qqq - 1;
    ht->n_nodes = 0;
    ht->n_inode = 0;
    ht->qend    = 1;

    return ht;
}

void destroy_tree(HuffmanTree* ht)
{
    free(ht->pool);
    ht->pool = nullptr;
    free(ht->qqq);
    ht->qqq = nullptr;
    for (size_t i = 0; i < ht->state_num; i++) {
        if (ht->code[i] != nullptr) free(ht->code[i]);
    }
    free(ht->code);
    ht->code = nullptr;
    free(ht->cout);
    ht->cout = nullptr;
    free(ht);
}

node_list new_node(HuffmanTree* ht, size_t freq, uint32_t c, node_list a, node_list b)
{
    node_list n = ht->pool + ht->n_nodes++;
    if (freq) {
        n->c    = c;
        n->freq = freq;
        n->t    = 1;
    }
    else {
        n->left  = a;
        n->right = b;
        n->freq  = a->freq + b->freq;
        n->t     = 0;
    }
    return n;
}

/* priority queue */
void qinsert(HuffmanTree* ht, node_list n)
{
    int j, i = ht->qend++;
    while ((j = (i >> 1))) {
        if (ht->qq[j]->freq <= n->freq) break;
        ht->qq[i] = ht->qq[j], i = j;
    }
    ht->qq[i] = n;
}

node_list qremove(HuffmanTree* ht)
{
    int i, l;
    node_list n = ht->qq[i = 1];
    node_list p;
    if (ht->qend < 2) return 0;
    ht->qend--;
    ht->qq[i] = ht->qq[ht->qend];

    while ((l = (i << 1)) < ht->qend) {
        if (l + 1 < ht->qend && ht->qq[l + 1]->freq < ht->qq[l]->freq) l++;
        if (ht->qq[i]->freq > ht->qq[l]->freq) {
            p         = ht->qq[i];
            ht->qq[i] = ht->qq[l];
            ht->qq[l] = p;
            i         = l;
        }
        else {
            break;
        }
    }

    return n;
}

void build_code(HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2)
{
    if (n->t) {
        ht->code[n->c] = (uint64_t*)malloc(2 * sizeof(uint64_t));
        if (len <= 64) {
            (ht->code[n->c])[0] = out1 << (64 - len);
            (ht->code[n->c])[1] = out2;
        }
        else {
            (ht->code[n->c])[0] = out1;
            (ht->code[n->c])[1] = out2 << (128 - len);
        }
        ht->cout[n->c] = (uint8_t)len;
        return;
    }

    int index = len >> 6;
    if (index == 0) {
        out1 = out1 << 1;
        out1 = out1 | 0;
        build_code(ht, n->left,  len + 1, out1, 0);
        out1 = out1 | 1;
        build_code(ht, n->right, len + 1, out1, 0);
    }
    else {
        if (len % 64 != 0) out2 = out2 << 1;
        out2 = out2 | 0;
        build_code(ht, n->left,  len + 1, out1, out2);
        out2 = out2 | 1;
        build_code(ht, n->right, len + 1, out1, out2);
    }
}

template <typename H>
void phf_CPU_build_codebook_v1(uint32_t* ext_freq, uint16_t booklen, H* book)
{
    auto state_num = 2 * booklen;
    auto all_nodes = 2 * state_num;

    auto freq = new uint32_t[all_nodes];
    memset(freq, 0, sizeof(uint32_t) * all_nodes);
    memcpy(freq, ext_freq, sizeof(uint32_t) * booklen);

    auto tree = create_tree_serial(state_num);

    for (size_t i = 0; i < tree->all_nodes; i++) {
        if (freq[i]) qinsert(tree, new_node(tree, freq[i], i, 0, 0));
    }
    while (tree->qend > 2)
        qinsert(tree, new_node(tree, 0, 0, qremove(tree), qremove(tree)));

    phf_stack<node_t, sizeof(H)>::template inorder_traverse<H>(tree->qq[1], book);

    destroy_tree(tree);
    delete[] freq;
}

template void phf_CPU_build_codebook_v1<uint32_t>(uint32_t*, uint16_t, uint32_t*);
template void phf_CPU_build_codebook_v1<uint64_t>(uint32_t*, uint16_t, uint64_t*);
template void phf_CPU_build_codebook_v1<unsigned long long>(uint32_t*, uint16_t, unsigned long long*);
