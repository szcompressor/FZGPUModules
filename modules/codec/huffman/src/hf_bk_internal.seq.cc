/**
 * @file hfserial_book2.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-17
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <queue>
#include <stdexcept>
#include <string>

#include "hf_impl.hh"
#include "timer.hh"

// internal data structure

#define __PSZDBG__FATAL(CONST_CHAR) \
  throw std::runtime_error("\033[31m[fzmod::fatal]\033[0m " + std::string(CONST_CHAR));

#define NODE_STACK_TPL template <class NodeType, int Width>
#define NODE_STACK phf_stack<NodeType, Width>

NODE_STACK_TPL NodeType* NODE_STACK::top(NODE_STACK* s) { return s->_a[s->depth - 1]; }

NODE_STACK_TPL template <typename T>
void NODE_STACK::push(NODE_STACK* s, NodeType* n, T path, T len)
{
  if (s->depth + 1 <= NODE_STACK::MAX_DEPTH) {
    s->depth += 1;

    s->_a[s->depth - 1] = n;
    s->saved_path[s->depth - 1] = path;
    s->saved_length[s->depth - 1] = len;
  }
  else
    throw std::runtime_error(
        "[fz::err::hf::traverse_stack]: exceeding MAX_DEPTH, stack "
        "overflow.");
}

NODE_STACK_TPL template <typename T>
NodeType* NODE_STACK::pop(NODE_STACK* s, T* path_to_restore, T* length_to_restore)
{
  auto is_empty = [&](NODE_STACK* s) -> bool { return (s->depth == 0); };

  NodeType* n;

  if (is_empty(s)) {
    printf("Error: stack underflow, exiting...\n");
    return nullptr;
  }
  else {
    // TODO holding array -> __a
    n = s->_a[s->depth - 1];
    s->_a[s->depth - 1] = nullptr;

    *length_to_restore = s->saved_length[s->depth - 1];
    *path_to_restore = s->saved_path[s->depth - 1];
    s->depth -= 1;

    return n;
  }
}

NODE_STACK_TPL template <typename H>
void NODE_STACK::inorder_traverse(NodeType* root, H* book)
{
  auto is_empty = [&](NODE_STACK* s) -> bool { return (s->depth == 0); };
  using PW = HuffmanWord<sizeof(H)>;
  constexpr auto MAX_LEN = PW::FIELD_CODE;

  auto s = new NODE_STACK();
  auto p = root;

  bool done = 0;
  H out1 = 0, len = 0;

  while (not done and p != nullptr) {
    if (p->left or p->right) {
      push(s, p, out1, len);
      p = p->left;
      out1 <<= 1u;
      out1 |= 0u;
      len += 1;

      if (len > MAX_LEN) __PSZDBG__FATAL("exceeding max len: " + std::to_string(MAX_LEN));
    }
    else {
      uint32_t symbol = p->symbol;
      book[symbol] = out1;
      reinterpret_cast<PW*>(&book[symbol])->bitcount = len;

      if (not is_empty(s)) {
        p = pop(s, &out1, &len);
        p = p->right;
        out1 <<= 1u;
        out1 |= 1u;
        len += 1;

        if (len > MAX_LEN) __PSZDBG__FATAL("exceeding max len: " + std::to_string(MAX_LEN));
      }
      else
        done = true;
    }
  }

  delete s;
  /* end of function */
}

template class phf_stack<node_t, 4>;
template void phf_stack<node_t, 4>::inorder_traverse<uint32_t>(node_t*, uint32_t*);
template class phf_stack<node_t, 8>;
template void phf_stack<node_t, 8>::inorder_traverse<uint64_t>(node_t*, uint64_t*);
template void phf_stack<node_t, 8>::inorder_traverse<unsigned long long>(node_t*, unsigned long long*);

template class phf_stack<phf_node, 4>;
template void phf_stack<phf_node, 4>::inorder_traverse<uint32_t>(phf_node*, uint32_t*);
template class phf_stack<phf_node, 8>;
template void phf_stack<phf_node, 8>::inorder_traverse<uint64_t>(phf_node*, uint64_t*);
template void phf_stack<phf_node, 8>::inorder_traverse<unsigned long long>(phf_node*, unsigned long long*);

#undef NODE_STACK_TPL
#undef NODE_STACK