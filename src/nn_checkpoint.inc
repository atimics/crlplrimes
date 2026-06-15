static size_t checkpoint_metadata_v1_size(void)
{
    return sizeof(uint32_t) + 65u + 65u + 65u + 256u;
}

static size_t checkpoint_metadata_size(void)
{
    return checkpoint_metadata_v1_size() + 32u;
}

static int checkpoint_meta_copy_text(char *dst, size_t dst_size, const char *value)
{
    if (dst == NULL || dst_size == 0) {
        return 0;
    }
    const char *src = value == NULL ? "" : value;
    size_t len = strlen(src);
    if (len >= dst_size) {
        dst[0] = '\0';
        return 0;
    }
    memcpy(dst, src, len + 1u);
    return 1;
}

void nn_checkpoint_meta_init(NNCheckpointMeta *meta, uint32_t feature_encoder_version)
{
    if (meta == NULL) {
        return;
    }
    memset(meta, 0, sizeof(*meta));
    meta->checkpoint_version = NN_CHECKPOINT_VERSION;
    meta->feature_encoder_version = feature_encoder_version;
}

int nn_checkpoint_meta_set_parent_model_hash(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->parent_model_hash,
                                     sizeof(meta->parent_model_hash),
                                     value);
}

int nn_checkpoint_meta_set_training_trace_root(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->training_trace_root,
                                     sizeof(meta->training_trace_root),
                                     value);
}

int nn_checkpoint_meta_set_benchmark_report_hash(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->benchmark_report_hash,
                                     sizeof(meta->benchmark_report_hash),
                                     value);
}

int nn_checkpoint_meta_set_notes(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->notes, sizeof(meta->notes), value);
}

int nn_checkpoint_meta_set_feature_set(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->feature_set, sizeof(meta->feature_set), value);
}

static int write_checkpoint_metadata(FILE *fp, const NNCheckpointMeta *meta)
{
    NNCheckpointMeta local;
    memset(&local, 0, sizeof(local));
    if (meta != NULL) {
        local = *meta;
    }

    if (local.checkpoint_version == 0) {
        local.checkpoint_version = NN_CHECKPOINT_VERSION;
    }
    local.parent_model_hash[sizeof(local.parent_model_hash) - 1] = '\0';
    local.training_trace_root[sizeof(local.training_trace_root) - 1] = '\0';
    local.benchmark_report_hash[sizeof(local.benchmark_report_hash) - 1] = '\0';
    local.notes[sizeof(local.notes) - 1] = '\0';
    local.feature_set[sizeof(local.feature_set) - 1] = '\0';

    return write_u32_le(fp, local.feature_encoder_version) &&
           write_all(fp, local.parent_model_hash, sizeof(local.parent_model_hash)) &&
           write_all(fp, local.training_trace_root, sizeof(local.training_trace_root)) &&
           write_all(fp, local.benchmark_report_hash, sizeof(local.benchmark_report_hash)) &&
           write_all(fp, local.notes, sizeof(local.notes)) &&
           write_all(fp, local.feature_set, sizeof(local.feature_set));
}

static int read_checkpoint_metadata(FILE *fp,
                                    uint32_t checkpoint_version,
                                    uint64_t metadata_size,
                                    NNCheckpointMeta *out_meta)
{
    NNCheckpointMeta local;
    memset(&local, 0, sizeof(local));
    local.checkpoint_version = checkpoint_version;

    if (metadata_size != (uint64_t)checkpoint_metadata_v1_size() &&
        metadata_size != (uint64_t)checkpoint_metadata_size()) {
        return 0;
    }

    if (!read_u32_le(fp, &local.feature_encoder_version) ||
        !read_all(fp, local.parent_model_hash, sizeof(local.parent_model_hash)) ||
        !read_all(fp, local.training_trace_root, sizeof(local.training_trace_root)) ||
        !read_all(fp, local.benchmark_report_hash, sizeof(local.benchmark_report_hash)) ||
        !read_all(fp, local.notes, sizeof(local.notes))) {
        return 0;
    }
    if (metadata_size == (uint64_t)checkpoint_metadata_size() &&
        !read_all(fp, local.feature_set, sizeof(local.feature_set))) {
        return 0;
    }

    local.parent_model_hash[sizeof(local.parent_model_hash) - 1] = '\0';
    local.training_trace_root[sizeof(local.training_trace_root) - 1] = '\0';
    local.benchmark_report_hash[sizeof(local.benchmark_report_hash) - 1] = '\0';
    local.notes[sizeof(local.notes) - 1] = '\0';
    local.feature_set[sizeof(local.feature_set) - 1] = '\0';

    if (out_meta != NULL) {
        *out_meta = local;
    }
    return 1;
}

int nn_save_checkpoint(const NeuralNet *net, const char *path, const NNCheckpointMeta *meta)
{
    if (net == NULL || path == NULL) {
        return 0;
    }

    uint32_t checkpoint_version =
        meta != NULL && meta->checkpoint_version != 0 ? meta->checkpoint_version
                                                      : NN_CHECKPOINT_VERSION;
    if (checkpoint_version != NN_CHECKPOINT_VERSION) {
        return 0;
    }

    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        return 0;
    }

    int ok = write_all(fp, NN_CHECKPOINT_MAGIC, NN_CHECKPOINT_MAGIC_SIZE) &&
             write_u32_le(fp, checkpoint_version) && write_u32_le(fp, scalar_type_id()) &&
             write_u64_le(fp, (uint64_t)net->layer_count);

    for (size_t layer = 0; ok && layer < net->layer_count; ++layer) {
        ok = write_u64_le(fp, (uint64_t)net->sizes[layer]);
    }

    ok = ok && write_u32_le(fp, (uint32_t)net->hidden_activation) &&
         write_u32_le(fp, (uint32_t)net->output_activation) &&
         write_u32_le(fp, (uint32_t)net->loss);

    for (size_t layer = 1; ok && layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        if (previous != 0 && current > SIZE_MAX / previous) {
            ok = 0;
            break;
        }
        size_t weight_count = previous * current;
        ok = write_all(fp, net->weights[layer], weight_count * sizeof(*net->weights[layer])) &&
             write_all(fp, net->biases[layer], current * sizeof(*net->biases[layer]));
    }

    ok = ok && write_u64_le(fp, (uint64_t)checkpoint_metadata_size()) &&
         write_checkpoint_metadata(fp, meta);

    if (fclose(fp) != 0) {
        ok = 0;
    }

    return ok;
}

NeuralNet *nn_load_checkpoint(const char *path, NNCheckpointMeta *out_meta)
{
    if (path == NULL) {
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }

    char magic[NN_CHECKPOINT_MAGIC_SIZE];
    uint32_t checkpoint_version = 0;
    uint32_t scalar_type = 0;
    uint64_t layer_count64 = 0;
    size_t *sizes = NULL;
    NeuralNet *net = NULL;
    int ok = read_all(fp, magic, sizeof(magic)) &&
             memcmp(magic, NN_CHECKPOINT_MAGIC, sizeof(magic)) == 0 &&
             read_u32_le(fp, &checkpoint_version) &&
             checkpoint_version == NN_CHECKPOINT_VERSION &&
             read_u32_le(fp, &scalar_type) && scalar_type == scalar_type_id() &&
             read_u64_le(fp, &layer_count64) && layer_count64 <= (uint64_t)SIZE_MAX;

    size_t layer_count = (size_t)layer_count64;
    if (ok) {
        sizes = checked_calloc(layer_count, sizeof(*sizes));
        ok = sizes != NULL;
    }

    for (size_t layer = 0; ok && layer < layer_count; ++layer) {
        uint64_t size64 = 0;
        ok = read_u64_le(fp, &size64) && size64 <= (uint64_t)SIZE_MAX;
        sizes[layer] = (size_t)size64;
    }

    uint32_t hidden_activation = 0;
    uint32_t output_activation = 0;
    uint32_t loss = 0;
    ok = ok && read_u32_le(fp, &hidden_activation) &&
         read_u32_le(fp, &output_activation) && read_u32_le(fp, &loss);

    if (ok) {
        NNConfig config;
        config.layer_sizes = sizes;
        config.layer_count = layer_count;
        config.seed = 1u;
        config.hidden_activation = (NNActivation)hidden_activation;
        config.output_activation = (NNActivation)output_activation;
        config.loss = (NNLoss)loss;
        net = nn_create_config(&config);
        ok = net != NULL;
    }

    for (size_t layer = 1; ok && layer < layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        if (previous != 0 && current > SIZE_MAX / previous) {
            ok = 0;
            break;
        }
        size_t weight_count = previous * current;
        ok = read_all(fp, net->weights[layer], weight_count * sizeof(*net->weights[layer])) &&
             read_all(fp, net->biases[layer], current * sizeof(*net->biases[layer]));
    }

    uint64_t metadata_size = 0;
    ok = ok && read_u64_le(fp, &metadata_size) &&
         read_checkpoint_metadata(fp, checkpoint_version, metadata_size, out_meta);

    if (fclose(fp) != 0) {
        ok = 0;
    }

    free(sizes);
    if (!ok) {
        nn_free(net);
        return NULL;
    }

    return net;
}

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char buffer[64];
    size_t buffer_count;
} NNSha256;

static uint32_t sha256_rotr(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static uint32_t sha256_load_be32(const unsigned char *bytes)
{
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

static void sha256_store_be32(unsigned char *bytes, uint32_t value)
{
    bytes[0] = (unsigned char)((value >> 24) & 0xffu);
    bytes[1] = (unsigned char)((value >> 16) & 0xffu);
    bytes[2] = (unsigned char)((value >> 8) & 0xffu);
    bytes[3] = (unsigned char)(value & 0xffu);
}

static void sha256_transform(NNSha256 *sha, const unsigned char block[64])
{
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t w[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;

    for (size_t i = 0; i < 16; ++i) {
        w[i] = sha256_load_be32(&block[i * 4]);
    }
    for (size_t i = 16; i < 64; ++i) {
        uint32_t s0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = sha256_rotr(w[i - 2], 17) ^ sha256_rotr(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    a = sha->state[0];
    b = sha->state[1];
    c = sha->state[2];
    d = sha->state[3];
    e = sha->state[4];
    f = sha->state[5];
    g = sha->state[6];
    h = sha->state[7];

    for (size_t i = 0; i < 64; ++i) {
        uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    sha->state[0] += a;
    sha->state[1] += b;
    sha->state[2] += c;
    sha->state[3] += d;
    sha->state[4] += e;
    sha->state[5] += f;
    sha->state[6] += g;
    sha->state[7] += h;
}

static void sha256_init(NNSha256 *sha)
{
    sha->state[0] = 0x6a09e667u;
    sha->state[1] = 0xbb67ae85u;
    sha->state[2] = 0x3c6ef372u;
    sha->state[3] = 0xa54ff53au;
    sha->state[4] = 0x510e527fu;
    sha->state[5] = 0x9b05688cu;
    sha->state[6] = 0x1f83d9abu;
    sha->state[7] = 0x5be0cd19u;
    sha->bit_count = 0;
    sha->buffer_count = 0;
}

static void sha256_update(NNSha256 *sha, const unsigned char *data, size_t size)
{
    sha->bit_count += (uint64_t)size * 8u;

    while (size > 0) {
        size_t available = sizeof(sha->buffer) - sha->buffer_count;
        size_t take = size < available ? size : available;
        memcpy(&sha->buffer[sha->buffer_count], data, take);
        sha->buffer_count += take;
        data += take;
        size -= take;

        if (sha->buffer_count == sizeof(sha->buffer)) {
            sha256_transform(sha, sha->buffer);
            sha->buffer_count = 0;
        }
    }
}

static void sha256_final(NNSha256 *sha, uint8_t out_hash[32])
{
    uint64_t bit_count = sha->bit_count;
    unsigned char pad = 0x80u;
    unsigned char zero = 0;
    unsigned char length[8];

    sha256_update(sha, &pad, 1);
    while (sha->buffer_count != 56) {
        sha256_update(sha, &zero, 1);
    }

    for (size_t i = 0; i < sizeof(length); ++i) {
        length[7 - i] = (unsigned char)((bit_count >> (8u * i)) & 0xffu);
    }
    sha256_update(sha, length, sizeof(length));

    for (size_t i = 0; i < 8; ++i) {
        sha256_store_be32(&out_hash[i * 4], sha->state[i]);
    }
}

int nn_hash_checkpoint_file(const char *path, uint8_t out_hash[32])
{
    if (path == NULL || out_hash == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return 0;
    }

    NNSha256 sha;
    sha256_init(&sha);

    unsigned char buffer[4096];
    int ok = 1;
    for (;;) {
        size_t count = fread(buffer, 1, sizeof(buffer), fp);
        if (count != 0) {
            sha256_update(&sha, buffer, count);
        }
        if (count < sizeof(buffer)) {
            if (ferror(fp)) {
                ok = 0;
            }
            break;
        }
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        return 0;
    }

    sha256_final(&sha, out_hash);
    return 1;
}

int nn_hash_checkpoint_hex(const char *path, char out_hex[65])
{
    static const char hex[] = "0123456789abcdef";
    uint8_t hash[32];

    if (out_hex == NULL || !nn_hash_checkpoint_file(path, hash)) {
        return 0;
    }

    for (size_t i = 0; i < sizeof(hash); ++i) {
        out_hex[i * 2] = hex[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex[hash[i] & 0x0fu];
    }
    out_hex[64] = '\0';
    return 1;
}
