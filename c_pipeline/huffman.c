#include "huffman.h"
#include "4dgs_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- byte_reader ---------------- */

int byte_reader_take(byte_reader_t *r, size_t n, const uint8_t **out) {
    if (r->offset + n > r->size) return -1;
    *out = r->data + r->offset;
    r->offset += n;
    return 0;
}
int byte_reader_u8(byte_reader_t *r, uint8_t *out) {
    const uint8_t *p; if (byte_reader_take(r, 1, &p) != 0) return -1;
    *out = p[0]; return 0;
}
int byte_reader_u16(byte_reader_t *r, uint16_t *out) {
    const uint8_t *p; if (byte_reader_take(r, 2, &p) != 0) return -1;
    *out = (uint16_t)(p[0] | (p[1] << 8)); return 0;
}
int byte_reader_u32(byte_reader_t *r, uint32_t *out) {
    const uint8_t *p; if (byte_reader_take(r, 4, &p) != 0) return -1;
    *out = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return 0;
}
int byte_reader_f32(byte_reader_t *r, float *out) {
    const uint8_t *p; if (byte_reader_take(r, 4, &p) != 0) return -1;
    memcpy(out, p, 4); return 0;
}

/* ---------------- byte_writer ---------------- */

void byte_writer_init(byte_writer_t *w) {
    w->data = NULL; w->size = 0; w->capacity = 0;
}
void byte_writer_free(byte_writer_t *w) {
    free(w->data); w->data = NULL; w->size = 0; w->capacity = 0;
}
static int bw_reserve(byte_writer_t *w, size_t need) {
    if (w->size + need <= w->capacity) return 0;
    size_t cap = w->capacity ? w->capacity * 2 : 1024;
    while (cap < w->size + need) cap *= 2;
    uint8_t *nd = (uint8_t *)realloc(w->data, cap);
    if (!nd) return -1;
    w->data = nd; w->capacity = cap;
    return 0;
}
int byte_writer_append(byte_writer_t *w, const void *src, size_t n) {
    if (bw_reserve(w, n) != 0) return -1;
    memcpy(w->data + w->size, src, n);
    w->size += n;
    return 0;
}
int byte_writer_u8(byte_writer_t *w, uint8_t v) {
    return byte_writer_append(w, &v, 1);
}
int byte_writer_u16(byte_writer_t *w, uint16_t v) {
    uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) };
    return byte_writer_append(w, b, 2);
}
int byte_writer_u32(byte_writer_t *w, uint32_t v) {
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8),
                     (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    return byte_writer_append(w, b, 4);
}
int byte_writer_f32(byte_writer_t *w, float v) {
    return byte_writer_append(w, &v, 4);
}

/* ---------------- decode tree ---------------- */

typedef struct {
    int child[2]; /* -1 = empty */
    int symbol;   /* -1 internal, else terminal */
} hnode_t;

int huffman_decode_block(byte_reader_t *r,
                         uint8_t **out_values, size_t *out_count) {
    uint16_t entry_count = 0;
    if (byte_reader_u16(r, &entry_count) != 0) return -1;

    /* Worst-case node count = sum(bit_len)+1; cap at a safe bound. */
    size_t cap_nodes = (size_t)entry_count * 64 + 8;
    hnode_t *tree = (hnode_t *)malloc(cap_nodes * sizeof(hnode_t));
    if (!tree) return -1;
    for (size_t i = 0; i < cap_nodes; ++i) {
        tree[i].child[0] = tree[i].child[1] = -1;
        tree[i].symbol = -1;
    }
    int node_count = 1;

    for (uint16_t i = 0; i < entry_count; ++i) {
        uint16_t sym;
        uint8_t  blen;
        uint32_t code;
        if (byte_reader_u16(r, &sym) != 0 ||
            byte_reader_u8 (r, &blen) != 0 ||
            byte_reader_u32(r, &code) != 0) {
            free(tree);
            return -1;
        }
        int node = 0;
        for (uint8_t b = 0; b < blen; ++b) {
            int bit = (code >> (blen - b - 1)) & 1;
            if (tree[node].child[bit] == -1) {
                if ((size_t)node_count >= cap_nodes) {
                    size_t new_cap = cap_nodes * 2;
                    hnode_t *grown = (hnode_t *)realloc(tree, new_cap * sizeof(hnode_t));
                    if (!grown) { free(tree); return -1; }
                    for (size_t k = cap_nodes; k < new_cap; ++k) {
                        grown[k].child[0] = grown[k].child[1] = -1;
                        grown[k].symbol = -1;
                    }
                    tree = grown; cap_nodes = new_cap;
                }
                tree[node].child[bit] = node_count++;
            }
            node = tree[node].child[bit];
        }
        tree[node].symbol = (sym == FOURDGS_HUFFMAN_EOF_SYMBOL)
            ? FOURDGS_HUFFMAN_EOF_INTERN : (int)sym;
    }

    uint32_t data_len = 0;
    const uint8_t *payload = NULL;
    if (byte_reader_u32(r, &data_len) != 0 ||
        byte_reader_take(r, data_len, &payload) != 0) {
        free(tree);
        return -1;
    }

    size_t out_cap = 1024;
    size_t out_n = 0;
    uint8_t *out = (uint8_t *)malloc(out_cap);
    if (!out) { free(tree); return -1; }

    int node = 0;
    for (uint32_t i = 0; i < data_len; ++i) {
        uint8_t byte = payload[i];
        for (int mask = 128; mask >= 1; mask >>= 1) {
            int bit = (byte & mask) ? 1 : 0;
            int next = tree[node].child[bit];
            if (next < 0) {
                free(tree); free(out);
                return -1;
            }
            node = next;
            if (tree[node].symbol >= 0) {
                int sym = tree[node].symbol;
                if (sym == FOURDGS_HUFFMAN_EOF_INTERN) {
                    *out_values = out;
                    *out_count = out_n;
                    free(tree);
                    return 0;
                }
                if (out_n == out_cap) {
                    out_cap *= 2;
                    uint8_t *grown = (uint8_t *)realloc(out, out_cap);
                    if (!grown) { free(tree); free(out); return -1; }
                    out = grown;
                }
                out[out_n++] = (uint8_t)sym;
                node = 0;
            }
        }
    }
    /* No EOF seen but payload fully consumed -> still return whatever decoded */
    *out_values = out;
    *out_count = out_n;
    free(tree);
    return 0;
}

/* ---------------- canonical encoder ---------------- */

/* Symbol space includes 256 EOF -> total 257. */
#define SYM_SPACE 257

typedef struct {
    int symbol;       /* index 0..256 */
    uint64_t freq;
    int parent;
    int left;
    int right;
} pq_node_t;

static int huffman_assign_lengths(const uint64_t *freq, uint8_t *bit_len) {
    /* Build canonical Huffman lengths via priority queue. */
    int active = 0;
    for (int i = 0; i < SYM_SPACE; ++i)
        if (freq[i] > 0) ++active;
    if (active == 0) return 0;

    pq_node_t *nodes = (pq_node_t *)calloc((size_t)active * 2, sizeof(pq_node_t));
    if (!nodes) return -1;
    int n_nodes = 0;
    for (int i = 0; i < SYM_SPACE; ++i) {
        if (freq[i] > 0) {
            nodes[n_nodes].symbol = i;
            nodes[n_nodes].freq = freq[i];
            nodes[n_nodes].parent = -1;
            nodes[n_nodes].left = nodes[n_nodes].right = -1;
            ++n_nodes;
        }
    }

    /* Heap of indices into nodes[]. Min by (freq, symbol). */
    int *heap = (int *)malloc(sizeof(int) * (size_t)(n_nodes * 2));
    if (!heap) { free(nodes); return -1; }
    int heap_n = 0;
    for (int i = 0; i < n_nodes; ++i) heap[heap_n++] = i;

#define LESS(a,b) (nodes[(a)].freq < nodes[(b)].freq || \
                   (nodes[(a)].freq == nodes[(b)].freq && nodes[(a)].symbol < nodes[(b)].symbol))

    /* Sift to heap */
    for (int i = heap_n / 2 - 1; i >= 0; --i) {
        int pos = i;
        for (;;) {
            int l = 2*pos+1, rr = 2*pos+2, best = pos;
            if (l < heap_n  && LESS(heap[l],  heap[best])) best = l;
            if (rr < heap_n && LESS(heap[rr], heap[best])) best = rr;
            if (best == pos) break;
            int tmp = heap[pos]; heap[pos] = heap[best]; heap[best] = tmp;
            pos = best;
        }
    }

    while (heap_n > 1) {
        int a = heap[0];
        heap[0] = heap[--heap_n];
        /* sift-down */
        int pos = 0;
        while (1) {
            int l = 2*pos+1, rr = 2*pos+2, best = pos;
            if (l < heap_n  && LESS(heap[l],  heap[best])) best = l;
            if (rr < heap_n && LESS(heap[rr], heap[best])) best = rr;
            if (best == pos) break;
            int tmp = heap[pos]; heap[pos] = heap[best]; heap[best] = tmp;
            pos = best;
        }
        int b = heap[0];
        heap[0] = heap[--heap_n];
        pos = 0;
        while (1) {
            int l = 2*pos+1, rr = 2*pos+2, best = pos;
            if (l < heap_n  && LESS(heap[l],  heap[best])) best = l;
            if (rr < heap_n && LESS(heap[rr], heap[best])) best = rr;
            if (best == pos) break;
            int tmp = heap[pos]; heap[pos] = heap[best]; heap[best] = tmp;
            pos = best;
        }

        /* Combine */
        int idx = n_nodes++;
        nodes[idx].symbol = -1;
        nodes[idx].freq = nodes[a].freq + nodes[b].freq;
        nodes[idx].parent = -1;
        nodes[idx].left = a;
        nodes[idx].right = b;
        nodes[a].parent = idx;
        nodes[b].parent = idx;

        /* Push */
        heap[heap_n] = idx;
        pos = heap_n;
        ++heap_n;
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (LESS(heap[pos], heap[parent])) {
                int tmp = heap[pos]; heap[pos] = heap[parent]; heap[parent] = tmp;
                pos = parent;
            } else break;
        }
    }

#undef LESS
    free(heap);

    /* Walk back up from each leaf to compute bit_len. */
    int root = n_nodes - 1;
    /* Edge case: only one symbol -> assign length 1. */
    int leaf_count = 0;
    for (int i = 0; i < n_nodes; ++i) {
        if (nodes[i].left == -1 && nodes[i].right == -1) leaf_count++;
    }
    if (leaf_count == 1) {
        for (int i = 0; i < n_nodes; ++i) {
            if (nodes[i].left == -1 && nodes[i].right == -1) {
                bit_len[nodes[i].symbol] = 1;
            }
        }
        free(nodes);
        return 0;
    }
    (void)root;
    for (int i = 0; i < n_nodes; ++i) {
        if (nodes[i].left != -1 || nodes[i].right != -1) continue;
        int depth = 0;
        int p = nodes[i].parent;
        while (p != -1) { depth++; p = nodes[p].parent; }
        if (depth > 32) depth = 32;  /* code_bits is uint32 */
        bit_len[nodes[i].symbol] = (uint8_t)depth;
    }
    free(nodes);
    return 0;
}

int huffman_build_table(const uint8_t *src, size_t n, huff_table_t *out) {
    uint64_t freq[SYM_SPACE] = {0};
    for (size_t i = 0; i < n; ++i) freq[src[i]]++;
    freq[FOURDGS_HUFFMAN_EOF_INTERN] = 1; /* always include EOF */

    uint8_t bit_len[SYM_SPACE] = {0};
    if (huffman_assign_lengths(freq, bit_len) != 0) return -1;

    /* Canonical codes: sort by (bit_len, symbol). */
    int order[SYM_SPACE];
    int active = 0;
    for (int i = 0; i < SYM_SPACE; ++i) {
        if (bit_len[i] > 0) order[active++] = i;
    }
    /* Sort by (bit_len, symbol). */
    for (int i = 1; i < active; ++i) {
        int x = order[i];
        int j = i - 1;
        while (j >= 0) {
            int y = order[j];
            int yk = (bit_len[y] << 9) | y;
            int xk = (bit_len[x] << 9) | x;
            if (yk > xk) {
                order[j + 1] = y;
                --j;
            } else break;
        }
        order[j + 1] = x;
    }

    out->count = (uint16_t)active;
    out->entries = (huff_entry_t *)calloc((size_t)active, sizeof(huff_entry_t));
    if (!out->entries) return -1;

    uint64_t code = 0;
    int prev_len = 0;
    for (int i = 0; i < active; ++i) {
        int s = order[i];
        int len = bit_len[s];
        if (i == 0) {
            code = 0;
        } else {
            code = (code + 1) << (len - prev_len);
        }
        prev_len = len;
        out->entries[i].symbol  = (s == FOURDGS_HUFFMAN_EOF_INTERN)
            ? FOURDGS_HUFFMAN_EOF_SYMBOL : (uint16_t)s;
        out->entries[i].bit_len = (uint8_t)len;
        out->entries[i].code_bits = (uint32_t)code;
    }
    return 0;
}

void huffman_table_free(huff_table_t *t) {
    free(t->entries);
    t->entries = NULL;
    t->count = 0;
}

int huffman_encode_block(byte_writer_t *w,
                         const uint8_t *src, size_t n,
                         const huff_table_t *table) {
    /* Write table */
    if (byte_writer_u16(w, table->count) != 0) return -1;
    for (uint16_t i = 0; i < table->count; ++i) {
        if (byte_writer_u16(w, table->entries[i].symbol)   != 0) return -1;
        if (byte_writer_u8 (w, table->entries[i].bit_len)  != 0) return -1;
        if (byte_writer_u32(w, table->entries[i].code_bits)!= 0) return -1;
    }

    /* Build symbol -> (bit_len, code_bits) lookup */
    uint8_t  blen[SYM_SPACE] = {0};
    uint32_t code[SYM_SPACE] = {0};
    for (uint16_t i = 0; i < table->count; ++i) {
        int s = table->entries[i].symbol == FOURDGS_HUFFMAN_EOF_SYMBOL
            ? FOURDGS_HUFFMAN_EOF_INTERN : table->entries[i].symbol;
        blen[s] = table->entries[i].bit_len;
        code[s] = table->entries[i].code_bits;
    }

    /* Pack bits MSB-first into a payload buffer. */
    size_t cap = n + 64;
    uint8_t *payload = (uint8_t *)malloc(cap);
    if (!payload) return -1;
    size_t bytes_used = 0;
    uint8_t cur = 0;
    int bit_in_cur = 0; /* 0..7, MSB first means we OR (1 << (7 - bit_in_cur)) */

    /* helper to push one symbol */
    #define PUSH_SYMBOL(s) do { \
        int _s = (s); \
        int  _len = blen[_s]; \
        uint32_t _c = code[_s]; \
        if (_len == 0) { free(payload); return -1; } \
        for (int _b = 0; _b < _len; ++_b) { \
            int _bit = (_c >> (_len - _b - 1)) & 1; \
            cur |= (uint8_t)(_bit << (7 - bit_in_cur)); \
            bit_in_cur++; \
            if (bit_in_cur == 8) { \
                if (bytes_used == cap) { \
                    cap *= 2; \
                    uint8_t *_g = (uint8_t *)realloc(payload, cap); \
                    if (!_g) { free(payload); return -1; } \
                    payload = _g; \
                } \
                payload[bytes_used++] = cur; \
                cur = 0; bit_in_cur = 0; \
            } \
        } \
    } while (0)

    for (size_t i = 0; i < n; ++i) PUSH_SYMBOL(src[i]);
    PUSH_SYMBOL(FOURDGS_HUFFMAN_EOF_INTERN);

    if (bit_in_cur > 0) {
        if (bytes_used == cap) {
            cap *= 2;
            uint8_t *g = (uint8_t *)realloc(payload, cap);
            if (!g) { free(payload); return -1; }
            payload = g;
        }
        payload[bytes_used++] = cur;
    }

    if (byte_writer_u32(w, (uint32_t)bytes_used) != 0) {
        free(payload); return -1;
    }
    if (byte_writer_append(w, payload, bytes_used) != 0) {
        free(payload); return -1;
    }
    free(payload);
    return 0;

    #undef PUSH_SYMBOL
}
