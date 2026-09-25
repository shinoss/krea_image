/* C API of the Krea 2 Turbo Metal engine (libkrea.dylib). */
#ifndef KREA_H
#define KREA_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct krea_engine krea_engine;

/* Live preview attached to "denoise" progress callbacks (valid only during the callback). */
typedef struct {
  int width, height;     /* preview size in pixels */
  int channels;          /* 3 = RGB (fast latent projection), 4 = RGBA (VAE decode) */
  int step;              /* denoising steps completed */
  int hq;                /* 0: fast 1/8-resolution latent projection, 1: full-resolution VAE decode */
  const uint8_t* x0;     /* predicted clean image  x0 = z - sigma * v */
  const uint8_t* noisy;  /* current noisy latent z (fast previews only, else NULL) */
} krea_preview;

/* Progress callback: stage is "loading", "encode", "denoise", "decode" or "done"; for "denoise",
   step = steps completed. preview may be NULL. Return non-zero to cancel. */
typedef int (*krea_progress_fn)(void* user, const char* stage, int step, int total, double elapsed_ms,
                                const krea_preview* preview);

typedef struct {
  int width, height;   /* pixels, multiples of 16 (rounded down) */
  int steps;           /* denoising steps (8 = Quality, the official Turbo setting; 4 with fast = 1) */
  uint64_t seed;
  float cfg_scale;     /* Krea convention: v = c + s (c - u); 0 disables (Turbo default) */
  int ane;             /* 1: offload part of every DiT layer to the Neural Engine (if built) */
  int preview;         /* 1: fast latent->RGB preview after every step (CPU only, overlaps the GPU) */
  int hq_preview_every;/* N > 0: also VAE-decode the predicted image every N steps (costs GPU time) */
  float* latent_out;   /* optional: receives the final packed latent, [H/16 * W/16, 64] f32 */
  int fast;            /* 1: 4-step distillation LoRA (merged weights); use steps = 4 */
  /* Editing (image-to-image / inpainting; Krea 2 has no instruction-editing mode). init_rgba: the source image,
     width*height*4 bytes (NULL = text-to-image). It is VAE-encoded and noised to sigma[start_step] (1 ..
     steps-1: a later start keeps more of it; 0 = start from pure noise), then denoised from there. mask:
     width*height bytes, >= 128 = regenerate, NULL (or empty) = the whole image. Outside the mask the source
     latent is re-imposed after every step and the source pixels are composited back with a feathered seam. */
  const uint8_t* init_rgba;
  const uint8_t* mask;
  int start_step;
} krea_params;

typedef struct {
  double encode_ms, denoise_ms, decode_ms, total_ms;
  double step_ms;      /* mean per-step time */
  int text_tokens;
  int cached_prompt;   /* 1 if the text path came from the prompt cache */
} krea_stats;

/* root: project directory containing weights/engine/ and engine/build/krea.metallib (env KREA_BUILD=<dir>
   reads engine/<dir>/krea.metallib instead; KREA_CACHE_DIR moves the prompt cache from <root>/cache/prompts). */
krea_engine* krea_create(const char* root, char* err, int errlen);
void krea_destroy(krea_engine* e);
int krea_ane_available(krea_engine* e);      /* GPU + Neural Engine weights installed */
int krea_fast_available(krea_engine* e);     /* 4-step LoRA weights installed */
int krea_gpu_only_available(krea_engine* e); /* full GPU-only DiT installed */

/* ids / neg_ids: token ids of [system prefix + prompt] + suffix (see ui/krea.py). out_rgba must hold
   width*height*4 bytes. Returns 0 on success, 1 if cancelled, -1 on error. */
int krea_generate(krea_engine* e, const int* ids, int n_ids, const int* neg_ids, int n_neg, const krea_params* p,
                  uint8_t* out_rgba, krea_stats* stats, krea_progress_fn cb, void* user, char* err, int errlen);

/* Encode a prompt ahead of krea_generate (e.g. while the user types): text encoder + text fusion, kept in the
   prompt cache under the same key. Returns 0 if encoded now, 1 if already cached, -1 on error. */
int krea_prepare_prompt(krea_engine* e, const int* ids, int n_ids, char* err, int errlen);

#ifdef __cplusplus
}
#endif
#endif
