/**
 * @file WebAssembly transcription of the neural.js inference engine.
 *
 * MEASUREMENT PROTOTYPE - nothing here ships; neural.js stays the
 * production engine. The point of this file is to answer "how much
 * faster could the client be" without giving up determinism.
 *
 * DETERMINISM BY FAITHFUL TRANSCRIPTION: WebAssembly f64 arithmetic
 * (+ - * / sqrt, floor) is IEEE-754 correctly rounded, exactly like
 * JavaScript's. Every routine below performs the SAME operations in
 * the SAME order as its neural.js counterpart, so results are
 * bit-identical - verified against every pinned vector by
 * wasm/wasm.test.mjs and wasm/verify.mjs. Rules observed:
 *
 *  - no libm, no -ffast-math, no reassociation, no FMA (WASM has no
 *    scalar FMA; the SIMD mul/add used below are separate, exactly
 *    rounded instructions - NEVER replace them with relaxed-simd);
 *  - detExp/detLog2 equivalents are transcribed term by term;
 *  - the v3 matmul4 kernel maps its four interleaved accumulators
 *    s0..s3 onto two f64x2 vector accumulators (lanes s0,s1 and
 *    s2,s3), four elements per iteration, combined (s0+s1)+(s2+s3) -
 *    the identical summation order, just two sums per instruction;
 *  - v1/v2 models keep the sequential single-accumulator kernel,
 *    which cannot be vectorized without reordering, so it stays
 *    scalar.
 *
 * Memory management is deliberately trivial: the JS glue (engine.js)
 * owns a bump allocator above __heap_base and passes byte offsets in.
 * Weights arrive as f32 (exactly as neural.js stores them - the f32
 * values are exact, math happens in f64), key/value cache entries are
 * f64 slabs, and per-session tables of k/v offsets let forked
 * sessions share cached positions by pointer exactly like the JS
 * engine's write-once arrays.
 */

#include <wasm_simd128.h>

#define MAX_DIM 1024
#define MAX_MLP 4096
#define MAX_VOCAB 4096
#define MAX_LEN 512
#define MAX_LAYERS 16

/* Exact double constants, identical literals to neural.js */
static const double LN2 = 0.6931471805599453;
static const double INV_LN2 = 1.4426950408889634;

/* POW2[i] = 2^(-i), built by halving (exact for normal doubles) */
static double POW2[1101];

/* Model descriptor, filled by init() */
static int g_dim, g_heads, g_layers, g_mlp, g_vocab, g_maxlen, g_use4;
static double g_attn_scale;

/* Tensor locations (byte offsets into linear memory, set by JS) */
static const float *t_embed, *t_pos, *t_norm;
static const float *t_norm1[MAX_LAYERS], *t_qkv[MAX_LAYERS],
  *t_proj[MAX_LAYERS], *t_norm2[MAX_LAYERS], *t_up[MAX_LAYERS],
  *t_down[MAX_LAYERS];

/* Scratch buffers (one inference step, mirrors the JS session's) */
static double s_x[MAX_DIM], s_h[MAX_DIM], s_qkv[3 * MAX_DIM],
  s_attn[MAX_DIM], s_proj[MAX_DIM], s_mlp[MAX_MLP],
  s_logits[MAX_VOCAB], s_scores[MAX_LEN];

/**
 * Deterministic exp() for non-positive arguments - transcribed from
 * neural.js detExp: reduction by ln2, then a fixed-order 13-term
 * Taylor series. floor lowers to the correctly-rounded f64.floor.
 */
static double det_exp (double x) {
  if (x < -708) return 0;
  double n = __builtin_floor(x * INV_LN2 + 0.5);
  double r = x - n * LN2;
  double term = 1;
  double sum = 1;
  for (int i = 1; i <= 13; i ++) {
    term = term * r / i;
    sum += term;
  }
  return POW2[-(int)n] * sum;
}

/** In-place softmax over x[0..n), transcribed from neural.js. */
static void softmax_n (double *x, int n) {
  double max = -__builtin_inf();
  for (int i = 0; i < n; i ++) {
    if (x[i] > max) max = x[i];
  }
  double sum = 0;
  for (int i = 0; i < n; i ++) {
    x[i] = det_exp(x[i] - max);
    sum += x[i];
  }
  for (int i = 0; i < n; i ++) x[i] /= sum;
}

/** RMS-norm of x scaled by g into out, transcribed from neural.js. */
static void rms_norm (const double *x, const float *g, double *out, int n) {
  double sum = 0;
  for (int i = 0; i < n; i ++) sum += x[i] * x[i];
  double scale = 1 / __builtin_sqrt(sum / n + 1e-5);
  for (int i = 0; i < n; i ++) out[i] = x[i] * scale * (double)g[i];
}

/**
 * Sequential matmul (v1/v2 kernel): one accumulator, strictly
 * left-to-right. The dependency chain forbids vectorization - the
 * summation order is frozen by issued v1/v2 links.
 */
static void matmul_seq (const float *w, int rows, int cols,
    const double *x, double *out) {
  for (int r = 0; r < rows; r ++) {
    double sum = 0;
    const float *base = w + (long)r * cols;
    for (int c = 0; c < cols; c ++) {
      sum += (double)base[c] * x[c];
    }
    out[r] = sum;
  }
}

/**
 * Unrolled matmul (v3 kernel): four interleaved partial sums combined
 * as (s0+s1)+(s2+s3), exactly neural.js matmul4. The two f64x2
 * accumulators hold lanes (s0,s1) and (s2,s3); each lane performs the
 * identical sequence of correctly-rounded mul/add operations the JS
 * scalars do, so the result is bit-identical while running four
 * MACs per iteration.
 */
static void matmul_4 (const float *w, int rows, int cols,
    const double *x, double *out) {
  for (int r = 0; r < rows; r ++) {
    const float *base = w + (long)r * cols;
    v128_t acc01 = wasm_f64x2_splat(0);
    v128_t acc23 = wasm_f64x2_splat(0);
    for (int c = 0; c < cols; c += 4) {
      v128_t wf = wasm_v128_load(base + c);
      v128_t w01 = wasm_f64x2_promote_low_f32x4(wf);
      v128_t w23 = wasm_f64x2_promote_low_f32x4(
        wasm_i32x4_shuffle(wf, wf, 2, 3, 2, 3));
      acc01 = wasm_f64x2_add(acc01, wasm_f64x2_mul(w01,
        wasm_v128_load(x + c)));
      acc23 = wasm_f64x2_add(acc23, wasm_f64x2_mul(w23,
        wasm_v128_load(x + c + 2)));
    }
    out[r] = (wasm_f64x2_extract_lane(acc01, 0)
        + wasm_f64x2_extract_lane(acc01, 1))
      + (wasm_f64x2_extract_lane(acc23, 0)
        + wasm_f64x2_extract_lane(acc23, 1));
  }
}

static void mm (const float *w, int rows, int cols,
    const double *x, double *out) {
  if (g_use4) matmul_4(w, rows, cols, x, out);
  else matmul_seq(w, rows, cols, x, out);
}

/**
 * Configures the engine for a loaded model. use4 selects the v3
 * kernel; v1/v2 models must pass 0 (their payloads froze the
 * sequential summation order).
 */
__attribute__((export_name("init")))
void init (int dim, int heads, int layers, int mlp, int vocab,
    int maxlen, int use4) {
  g_dim = dim;
  g_heads = heads;
  g_layers = layers;
  g_mlp = mlp;
  g_vocab = vocab;
  g_maxlen = maxlen;
  g_use4 = use4;
  g_attn_scale = 1 / __builtin_sqrt((double)(dim / heads));
  POW2[0] = 1;
  for (int i = 1; i < 1101; i ++) POW2[i] = POW2[i - 1] * 0.5;
}

/**
 * Registers a tensor's location. kind: 0 embed, 1 pos, 2 final norm,
 * 3 norm1, 4 qkv, 5 proj, 6 norm2, 7 up, 8 down (3..8 per layer).
 */
__attribute__((export_name("set_tensor")))
void set_tensor (int kind, int layer, unsigned offset) {
  const float *p = (const float *)offset;
  switch (kind) {
    case 0: t_embed = p; break;
    case 1: t_pos = p; break;
    case 2: t_norm = p; break;
    case 3: t_norm1[layer] = p; break;
    case 4: t_qkv[layer] = p; break;
    case 5: t_proj[layer] = p; break;
    case 6: t_norm2[layer] = p; break;
    case 7: t_up[layer] = p; break;
    case 8: t_down[layer] = p; break;
  }
}

/** Address of the probability output buffer (vocab doubles). */
__attribute__((export_name("probs_ptr")))
unsigned probs_ptr (void) {
  return (unsigned)s_logits;
}

/**
 * One transformer step, transcribed from the JS session's feed().
 * The table holds, per layer and cached position, the byte offsets
 * of that position's key and value vectors (dim doubles each):
 * table[(l * maxLen + j) * 2] = key, + 1 = value. The entries for
 * `position` point at freshly allocated slabs this call fills in -
 * the JS glue writes them before calling. After the call the probs
 * buffer holds softmax(logits), which is what both the coder and the
 * tokenization search consume.
 */
__attribute__((export_name("feed")))
void feed (int id, int position, unsigned table_off) {
  const unsigned *table = (const unsigned *)table_off;
  const int dim = g_dim;
  const int head_dim = dim / g_heads;
  const int count = position + 1;

  for (int i = 0; i < dim; i ++) {
    s_x[i] = (double)t_embed[id * dim + i] + (double)t_pos[position * dim + i];
  }

  for (int l = 0; l < g_layers; l ++) {
    // Attention
    rms_norm(s_x, t_norm1[l], s_h, dim);
    mm(t_qkv[l], 3 * dim, dim, s_h, s_qkv);
    const unsigned *row = table + (long)l * g_maxlen * 2;
    double *k_new = (double *)row[position * 2];
    double *v_new = (double *)row[position * 2 + 1];
    for (int i = 0; i < dim; i ++) {
      k_new[i] = s_qkv[dim + i];
      v_new[i] = s_qkv[2 * dim + i];
    }
    for (int head = 0; head < g_heads; head ++) {
      const int base = head * head_dim;
      for (int j = 0; j < count; j ++) {
        double sum = 0;
        const double *k = (const double *)row[j * 2];
        for (int i = 0; i < head_dim; i ++) {
          sum += s_qkv[base + i] * k[base + i];
        }
        s_scores[j] = sum * g_attn_scale;
      }
      softmax_n(s_scores, count);
      for (int i = 0; i < head_dim; i ++) s_attn[base + i] = 0;
      for (int j = 0; j < count; j ++) {
        const double *v = (const double *)row[j * 2 + 1];
        const double weight = s_scores[j];
        for (int i = 0; i < head_dim; i ++) {
          s_attn[base + i] += weight * v[base + i];
        }
      }
    }
    mm(t_proj[l], dim, dim, s_attn, s_proj);
    for (int i = 0; i < dim; i ++) s_x[i] += s_proj[i];

    // MLP
    rms_norm(s_x, t_norm2[l], s_h, dim);
    mm(t_up[l], g_mlp, dim, s_h, s_mlp);
    for (int i = 0; i < g_mlp; i ++) {
      if (s_mlp[i] < 0) s_mlp[i] = 0;
    }
    mm(t_down[l], dim, g_mlp, s_mlp, s_proj);
    for (int i = 0; i < dim; i ++) s_x[i] += s_proj[i];
  }

  rms_norm(s_x, t_norm, s_h, dim);
  // Output head is tied to the embedding table
  mm(t_embed, g_vocab, dim, s_h, s_logits);
  softmax_n(s_logits, g_vocab);
}
