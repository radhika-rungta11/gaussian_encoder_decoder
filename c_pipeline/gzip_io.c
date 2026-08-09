#include "gzip_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

int read_file_all(const char *path, uint8_t **out_data, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[gzip_io] cannot open %s\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return -1;
    }
    rewind(fp);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) {
        fclose(fp);
        return -1;
    }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *out_data = buf;
    *out_size = (size_t)sz;
    return 0;
}

int write_file_all(const char *path, const uint8_t *data, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    if (fwrite(data, 1, size, fp) != size) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

int gzip_inflate_if_needed(const uint8_t *src, size_t src_size,
                           uint8_t **out_data, size_t *out_size) {
    if (src_size < 2 || src[0] != 0x1F || src[1] != 0x8B) {
        uint8_t *copy = (uint8_t *)malloc(src_size);
        if (!copy) return -1;
        memcpy(copy, src, src_size);
        *out_data = copy;
        *out_size = src_size;
        return 0;
    }

    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) return -1;

    size_t cap = src_size * 8 + 1024;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) { inflateEnd(&strm); return -1; }

    strm.next_in = (Bytef *)src;
    strm.avail_in = (uInt)src_size;

    for (;;) {
        if (strm.total_out >= cap) {
            size_t new_cap = cap * 2;
            uint8_t *grown = (uint8_t *)realloc(buf, new_cap);
            if (!grown) { free(buf); inflateEnd(&strm); return -1; }
            buf = grown;
            cap = new_cap;
        }
        strm.next_out = buf + strm.total_out;
        strm.avail_out = (uInt)(cap - strm.total_out);
        int rc = inflate(&strm, Z_NO_FLUSH);
        if (rc == Z_STREAM_END) break;
        if (rc != Z_OK) {
            free(buf);
            inflateEnd(&strm);
            return -1;
        }
    }
    *out_size = strm.total_out;
    *out_data = buf;
    inflateEnd(&strm);
    return 0;
}

int gzip_deflate(const uint8_t *src, size_t src_size,
                 uint8_t **out_data, size_t *out_size) {
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                     16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        return -1;
    }
    size_t cap = deflateBound(&strm, (uLong)src_size) + 64;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) { deflateEnd(&strm); return -1; }

    strm.next_in   = (Bytef *)src;
    strm.avail_in  = (uInt)src_size;
    strm.next_out  = buf;
    strm.avail_out = (uInt)cap;

    if (deflate(&strm, Z_FINISH) != Z_STREAM_END) {
        free(buf);
        deflateEnd(&strm);
        return -1;
    }
    *out_size = strm.total_out;
    *out_data = buf;
    deflateEnd(&strm);
    return 0;
}
