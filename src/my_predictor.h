#include <string.h>
#include <stdint.h>
#include "branch.h"
#include "predictor.h"

/*------------------------------------------------------------
  Global Constants & Table Sizes
  These macros define the size, bit-width and mask for each
  predictor component: base predictor, TAGE banks, local history,
  loop predictor, side-channels, perceptron, and meta predictor.
------------------------------------------------------------*/
#define MAX_GHIST_LEN         64

// Base predictor: simple bimodal 2-bit counters
#define BASE_BITS             16
#define BASE_SIZE             (1 << BASE_BITS)
#define BASE_MASK             (BASE_SIZE - 1)

// TAGE configuration: number of banks and their history lengths
#define TAGE_BANKS            5
static const unsigned int TAGE_HIST_LEN[TAGE_BANKS] = { 4, 8, 16, 32, 64 };

#define TAGE_BANK_BITS        16
#define TAGE_BANK_SIZE        (1 << TAGE_BANK_BITS)
#define TAGE_BANK_MASK        (TAGE_BANK_SIZE - 1)

#define TAGE_TAG_BITS         16
#define TAGE_TAG_MASK         ((1 << TAGE_TAG_BITS) - 1)

// Local history predictor parameters
#define LOCAL_HISTORY_BITS        14
#define LOCAL_HISTORY_TABLE_BITS  16
#define LOCAL_HISTORY_TABLE_SIZE  (1 << LOCAL_HISTORY_TABLE_BITS)
#define LOCAL_HISTORY_TABLE_MASK  (LOCAL_HISTORY_TABLE_SIZE - 1)

#define LOCAL_PHT_BITS        LOCAL_HISTORY_BITS
#define LOCAL_PHT_SIZE        (1 << LOCAL_PHT_BITS)
#define LOCAL_PHT_MASK        (LOCAL_PHT_SIZE - 1)

// Chooser table for selecting TAGE or Local
#define CHOOSER_BITS          16
#define CHOOSER_SIZE          (1 << CHOOSER_BITS)
#define CHOOSER_MASK          ((1 << CHOOSER_BITS) - 1)

// Loop predictor configuration
#define LP_BITS               12
#define LP_SIZE               (1 << LP_BITS)
#define LP_MASK               ((1u << LP_BITS) - 1u)
#define LP_CONF_MAX           3
#define LP_STRONG_CONF        4
#define LP_PERIOD_MAX         512u

// Side-channel predictors (three small hashed sums)
#define SC_BITS               12
#define SC_SIZE               (1 << SC_BITS)
#define SC_MASK               ((1 << SC_BITS) - 1)

// Perceptron predictor configuration (global history features)
#define PERC_ROWS_BITS        10
#define PERC_ROWS             (1 << PERC_ROWS_BITS)
#define PERC_HIST             64
#define PERC_WMAX             127
#define PERC_TH               28
#define PERC_TH_TRAIN         20
#define PERC_TH_TIE           30

// Meta predictor (combines TAGE/Local/Perceptron)
#define META_ROWS_BITS        10
#define META_ROWS             (1 << META_ROWS_BITS)
#define META_FEATS            19
#define META_WMAX             127
#define META_TH               12
#define META_TH_TRAIN         8

/*------------------------------------------------------------
  Simple 2-bit counter helpers
  - ctr_pred: 2/3 => taken; 0/1 => not-taken
  - ctr_update: move counter one step toward the outcome
------------------------------------------------------------*/
static inline bool ctr_pred(unsigned char c) { return (c >> 1); }

static inline void ctr_update(unsigned char &c, bool taken) {
    if (taken) { if (c < 3) c++; }
    else       { if (c > 0) c--; }
}

/*------------------------------------------------------------
  TAGE Table Entry
  - tag: partial tag for PC/history match
  - ctr: 2-bit prediction counter
  - u:   usefulness/age counter for replacement decisions
------------------------------------------------------------*/
struct tage_entry_t {
    unsigned short tag;
    unsigned char  ctr;
    unsigned char  u;
};

/*------------------------------------------------------------
  my_update
  Carries all indices/predictions from predict() to update().
  This avoids recomputation and ensures consistent training.
------------------------------------------------------------*/
class my_update : public branch_update {
public:
    bool is_conditional;
    bool final_pred;

    // Local predictor state
    unsigned int local_hist_idx;
    unsigned int local_pht_idx;
    bool local_pred;

    // Chooser state
    unsigned int chooser_idx;

    // TAGE state
    bool tage_pred;
    bool base_pred;
    int  provider_bank;
    int  alt_bank;
    unsigned int base_idx;
    unsigned int bank_idx[TAGE_BANKS];
    bool bank_match[TAGE_BANKS];

    // Loop predictor snapshot
    bool loop_hit;
    bool loop_pred;
    unsigned char loop_conf;

    // Side-channel combined sum
    int sc_sum;

    // Perceptron state
    int  perc_row;
    int  perc_y;
    bool perc_pred;

    // Meta predictor state
    int  meta_row;
    int  meta_y;
    bool meta_choose_tage;
};

/*------------------------------------------------------------
  my_predictor
  A hybrid predictor composed of:
    - Base bimodal
    - Local history predictor
    - TAGE (multiple banks with different hist lengths)
    - Perceptron (global history)
    - Meta perceptron combiner
    - Loop predictor (periodic behavior)
    - Side-channel hashed sums (small extra bias)
------------------------------------------------------------*/
class my_predictor : public branch_predictor {
public:
    // Global history register (up to 64 bits used)
    unsigned long long ghr;

    // Base predictor: 2-bit counters
    unsigned char base_table[BASE_SIZE];

    // TAGE banks: each bank uses a different effective history length
    tage_entry_t tage_banks[TAGE_BANKS][TAGE_BANK_SIZE];

    // Local history-based predictor
    unsigned short local_history_table[LOCAL_HISTORY_TABLE_SIZE]; // per-PC local history bits
    unsigned char  local_pht[LOCAL_PHT_SIZE];                     // 2-bit counters indexed by local history

    // Chooser that selects TAGE vs Local when meta is uncertain
    unsigned char chooser[CHOOSER_SIZE];

    // Loop predictor entry captures periodic loop behavior
    struct LoopEntry { unsigned int pc_tag; unsigned short period, cnt; unsigned char conf; };
    LoopEntry loop_tbl[LP_SIZE];

    // Lightweight side-channel components (three hashed accumulators)
    int8_t sc1[SC_SIZE];
    int8_t sc2[SC_SIZE];
    int8_t sc3[SC_SIZE];

    // Perceptron: rows × (bias + PERC_HIST history features)
    int8_t perc[PERC_ROWS][PERC_HIST + 1];

    // Meta perceptron: rows × (bias + META_FEATS)
    // META_FEATS = 3 expert signs + 8 GH bits + 8 PC bits
    int8_t meta[META_ROWS][META_FEATS + 1];

    // Book-keeping for update()
    branch_info last_bi;
    my_update u;

    // RNG state (xorshift64)
    uint64_t rng;

    /*------------------------------------------------------------
      Constructor
      Initializes:
        - base/local/chooser: neutral/weakly taken
        - TAGE: cleared tags, weak counters, zero usefulness
        - loop/sc/perc/meta: zeroed
        - GHR and update scratch
    ------------------------------------------------------------*/
    my_predictor(void) {
        ghr = 0ull;

        // Base bimodal: 2 = weakly taken
        for (int i = 0; i < BASE_SIZE; i++) base_table[i] = 2;

        // TAGE banks: tag=0, ctr=2 (weakly taken), u=0
        for (int b = 0; b < TAGE_BANKS; b++) {
            for (int i = 0; i < TAGE_BANK_SIZE; i++) {
                tage_banks[b][i].tag = 0;
                tage_banks[b][i].ctr = 2;
                tage_banks[b][i].u   = 0;
            }
        }

        // Local history & PHT
        memset(local_history_table, 0, sizeof(local_history_table));
        for (int i = 0; i < LOCAL_PHT_SIZE; i++) local_pht[i] = 2; // weakly taken

        // Chooser starts near-neutral (1)
        for (int i = 0; i < CHOOSER_SIZE; i++) chooser[i] = 1;

        // Loop predictor reset
        for (int i = 0; i < LP_SIZE; i++) {
            loop_tbl[i].pc_tag = 0;
            loop_tbl[i].period = 0;
            loop_tbl[i].cnt    = 0;
            loop_tbl[i].conf   = 0;
        }

        // Side-channels
        memset(sc1, 0, sizeof(sc1));
        memset(sc2, 0, sizeof(sc2));
        memset(sc3, 0, sizeof(sc3));

        // Perceptron & Meta weights
        memset(perc, 0, sizeof(perc));
        memset(meta, 0, sizeof(meta));

        // Per-call update scratch
        //memset(&u, 0, sizeof(u));
        u = my_update{};

        // Non-zero seed
        rng = 0x9e3779b97f4a7c15ull;
    }

    /*------------------------------------------------------------
      xorshift64 RNG
      - Small, fast PRNG used for optional randomized behavior.
      - Ensures non-zero state by restoring seed if it becomes 0.
    ------------------------------------------------------------*/
    inline uint64_t xorshift64() {
        uint64_t x = rng;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        rng = x ? x : 0x9e3779b97f4a7c15ull;
        return rng;
    }

    /*------------------------------------------------------------
      Indexing helpers (PC/GHR hashing)
      Keep hashing simple and fast; masks enforce table bounds.
    ------------------------------------------------------------*/
    inline unsigned int get_base_idx(unsigned int pc) { return (pc >> 2) & BASE_MASK; }

    inline unsigned int get_local_hist_idx(unsigned int pc) { return (pc >> 2) & LOCAL_HISTORY_TABLE_MASK; }

    // Chooser index mixes PC bits and low GHR bits
    inline unsigned int get_chooser_idx(unsigned int pc) {
        unsigned int pc_low  = (pc >> 2) & CHOOSER_MASK;
        unsigned int ghr_low = (unsigned int)(ghr & CHOOSER_MASK);
        return (pc_low ^ ghr_low) & CHOOSER_MASK;
    }

    /*------------------------------------------------------------
      TAGE index & tag computation
      - Each bank uses a different history length.
      - Fold GHR bits into index/tag via XOR to decorrelate.
    ------------------------------------------------------------*/
    inline unsigned int tage_index(int b, unsigned int pc) {
        unsigned int idx = (pc >> 2);
        unsigned long long gh = ghr;
        unsigned int hlen = TAGE_HIST_LEN[b];
        unsigned int mix = 0;
        // Fold the first hlen bits of GHR into index positions
        for (unsigned int i = 0; i < hlen; i++) {
            mix ^= ((unsigned int)((gh >> i) & 1u) << (i % TAGE_BANK_BITS));
        }
        idx ^= mix;
        return idx & TAGE_BANK_MASK;
    }

    inline unsigned short tage_tag(int b, unsigned int pc) {
        unsigned int hlen = TAGE_HIST_LEN[b];
        unsigned long long gh = ghr;
        unsigned int tagmix = pc;
        // Fold the first hlen bits of GHR into the tag field
        for (unsigned int i = 0; i < hlen; i++) {
            tagmix ^= ((unsigned int)((gh >> i) & 1u) << (i % TAGE_TAG_BITS));
        }
        return (unsigned short)(tagmix & TAGE_TAG_MASK);
    }

    /*------------------------------------------------------------
      Loop predictor indexing & behavior
      - Tracks periodic patterns (e.g., loop taken until exit).
      - "hit" when pc_tag matches and a plausible period is learned.
    ------------------------------------------------------------*/
    inline unsigned lp_index(unsigned pc){ return (((pc>>2) ^ (pc>>13)) & LP_MASK); }

    void loop_predict(unsigned pc, bool conditional, bool &hit, bool &pred, unsigned char &conf) {
        hit=false; pred=false; conf=0;
        if(!conditional) return;                    // only for conditional branches
        LoopEntry &e = loop_tbl[lp_index(pc)];
        if (e.pc_tag == (pc>>2) && e.period>1) {    // learned a usable period
            hit = true;
            conf = e.conf;
            // Predict "taken" while still within the loop body
            pred = (e.cnt + 1 < e.period);
        }
    }

    // Online learning of loop period, count, and confidence
    void loop_update(unsigned pc, bool taken) {
        LoopEntry &e = loop_tbl[lp_index(pc)];
        if (e.pc_tag != (pc>>2)) {
            // Different PC mapped here: reset learning state
            e.pc_tag = (pc>>2);
            e.period=0; e.cnt=0; e.conf=0;
        }
        if (taken) {
            // Count iterations until a not-taken exit
            if (e.cnt < LP_PERIOD_MAX) e.cnt++;
            // Stretch the period if needed
            if (e.period && e.cnt > e.period) e.period = e.cnt;
        } else {
            // On exit, reinforce a stable period; otherwise decay confidence
            if (e.cnt>0) {
                if (e.period==0) e.period = e.cnt;
                else if (e.period==e.cnt && e.conf<LP_CONF_MAX) e.conf++;
            } else {
                if (e.conf>0) e.conf--;
            }
            e.cnt = 0;
        }
    }

    /*------------------------------------------------------------
      Side-channel indices
      - Three independent hashed sums add a small extra bias.
      - Trained as tiny saturating sums in update().
    ------------------------------------------------------------*/
    inline unsigned sc_idx1(unsigned pc){ return (((pc>>2) ^ (unsigned)ghr) & SC_MASK); }
    inline unsigned sc_idx2(unsigned pc){ return (((pc>>9) ^ (unsigned)((ghr>>7) ^ (ghr<<5))) & SC_MASK); }
    inline unsigned sc_idx3(unsigned pc){ return (((pc>>4) ^ (unsigned)((ghr>>13) ^ (ghr<<11))) & SC_MASK); }

    /*------------------------------------------------------------
      Perceptron row selection and evaluation
      - Row is a hash of PC and GHR.
      - Features = {bias=+1} ∪ {GH bit i mapped to ±1}.
      - Score y >= 0 → predict taken; else not-taken.
    ------------------------------------------------------------*/
    inline int perc_row_idx(unsigned pc){
        unsigned g = (unsigned)ghr;
        unsigned r = (pc >> 2) ^ (pc >> 13) ^ (pc * 0x9e3779b9u) ^ (g ^ (g >> 11));
        return (int)(r & (PERC_ROWS - 1));
    }

    inline int perc_eval_row(int row, unsigned long long gh){
        int y = (int)perc[row][0]; // bias weight
        for (int i = 0; i < PERC_HIST; i++) {
            int xi = ((gh >> i) & 1ull) ? +1 : -1;
            y += xi * (int)perc[row][i+1];
        }
        return y;
    }

    // Perceptron learning with margin and clipping
    inline void perc_train_row(int row, bool taken, int y, unsigned long long gh){
        int t = taken ? +1 : -1;
        // Train on mistakes or low-margin correct predictions
        if ((t * y) <= 0 || (y < PERC_TH_TRAIN && y > -PERC_TH_TRAIN)) {
            // Bias update
            int w0 = (int)perc[row][0] + t;
            if (w0 > PERC_WMAX) w0 = PERC_WMAX; else if (w0 < -PERC_WMAX) w0 = -PERC_WMAX;
            perc[row][0] = (int8_t)w0;
            // Feature weights update
            for (int i = 0; i < PERC_HIST; i++) {
                int xi = ((gh >> i) & 1ull) ? +1 : -1;
                int wi = (int)perc[row][i+1] + t * xi;
                if (wi > PERC_WMAX) wi = PERC_WMAX; else if (wi < -PERC_WMAX) wi = -PERC_WMAX;
                perc[row][i+1] = (int8_t)wi;
            }
        }
    }

    /*------------------------------------------------------------
      Meta predictor (small perceptron)
      - Features: {TAGE sign, Local sign, Perceptron sign} + 8 GH bits + 8 PC bits → 19 dims.
      - Output > 0 prefers TAGE; < 0 prefers Local (unless meta confidence is low).
    ------------------------------------------------------------*/
    inline int meta_row_idx(unsigned pc){
        unsigned g = (unsigned)ghr;
        unsigned r = (pc >> 2) ^ (pc >> 11) ^ (pc * 0x9e3779b9u) ^ (g ^ (g >> 7));
        return (int)(r & (META_ROWS - 1));
    }

    inline void meta_build_feats(bool tage_p, bool local_p, bool perc_p, int feats[META_FEATS], unsigned pc){
        // 0..2: signs of experts
        feats[0] = tage_p  ? +1 : -1;
        feats[1] = local_p ? +1 : -1;
        feats[2] = perc_p  ? +1 : -1;

        // 3..10: low GH bits → ±1
        unsigned g = (unsigned)ghr;
        for (int i=0;i<8;i++) feats[3+i]  = ((g>>i)&1u)       ? +1 : -1;

        // 11..18: low PC bits → ±1
        for (int i=0;i<8;i++) feats[11+i] = (((pc>>2)>>i)&1u) ? +1 : -1;
    }

    inline int meta_eval_row(int row, const int feats[META_FEATS]){
        int y = (int)meta[row][0]; // bias
        for (int i=0;i<META_FEATS;i++){
            y += feats[i] * (int)meta[row][i+1];
        }
        return y;
    }

    // Train meta on disagreements or near-margin cases
    inline void meta_train_row(int row, const int feats[META_FEATS], int target){
        int y = (int)meta[row][0];
        for (int i=0;i<META_FEATS;i++) y += feats[i] * (int)meta[row][i+1];

        if ((target * y) <= 0 || (y < META_TH_TRAIN && y > -META_TH_TRAIN)) {
            // Bias update toward target
            int b = (int)meta[row][0] + target;
            if (b > META_WMAX) b = META_WMAX; else if (b < -META_WMAX) b = -META_WMAX;
            meta[row][0] = (int8_t)b;

            // Feature weights update
            for (int i=0;i<META_FEATS;i++){
                int w = (int)meta[row][i+1] + target * feats[i];
                if (w > META_WMAX) w = META_WMAX; else if (w < -META_WMAX) w = -META_WMAX;
                meta[row][i+1] = (int8_t)w;
            }
        }
    }

    /*------------------------------------------------------------
      predict()
      Flow for conditional branches:
        1) Read base, local, and TAGE candidates.
        2) Resolve TAGE provider/alt bank; pick TAGE guess.
        3) Evaluate perceptron (global) and meta (combiner).
        4) Use chooser + meta confidence for final_guess.
        5) Compute loop/side-channel signals (for diagnostics/training).
        6) Store all info in my_update for update().
    ------------------------------------------------------------*/
    branch_update *predict (branch_info & binfo) {
        last_bi = binfo;
        unsigned int pc = binfo.address;
        bool conditional = (binfo.br_flags & BR_CONDITIONAL) != 0;
        u.is_conditional = conditional;

        // ---- Local predictor: read local history → index PHT → 2-bit counter
        unsigned int lhist_idx = get_local_hist_idx(pc);
        unsigned int lhist_val = local_history_table[lhist_idx] & LOCAL_PHT_MASK;
        unsigned int lpht_idx  = lhist_val;
        bool local_guess       = ctr_pred(local_pht[lpht_idx]);

        // ---- Base bimodal predictor
        unsigned int bidx      = get_base_idx(pc);
        bool base_guess        = ctr_pred(base_table[bidx]);

        // ---- TAGE: indices/tags for each bank and tag matches
        bool bank_match[TAGE_BANKS];
        unsigned int bank_idx[TAGE_BANKS];
        for (int bi = 0; bi < TAGE_BANKS; bi++) {
            bank_idx[bi] = tage_index(bi, pc);
            unsigned short need_tag = tage_tag(bi, pc);
            tage_entry_t &e = tage_banks[bi][bank_idx[bi]];
            bank_match[bi] = (e.tag == need_tag);
        }

        // ---- TAGE: select provider bank (longest matching history)
        int provider_bank = -1;
        int alt_bank = -1;
        bool tage_guess = base_guess;

        // Search from longer history (TAGE_BANKS-2) down to 0
        for (int bi = TAGE_BANKS - 2; bi >= 0; bi--) {
            if (bank_match[bi]) {
                provider_bank = bi;
                tage_guess = ctr_pred(tage_banks[bi][bank_idx[bi]].ctr);
                break;
            }
        }
        // Special case for the longest bank: require strong counter and some usefulness
        if (provider_bank < 0 && bank_match[TAGE_BANKS - 1]) {
            unsigned char c = tage_banks[TAGE_BANKS - 1][bank_idx[TAGE_BANKS - 1]].ctr;
            unsigned char u3= tage_banks[TAGE_BANKS - 1][bank_idx[TAGE_BANKS - 1]].u;
            if ((c==0 || c==3) && u3>=1) {
                provider_bank = TAGE_BANKS - 1;
                tage_guess = ctr_pred(c);
            }
        }

        // Determine alt_bank = next shorter matching bank under provider
        if (provider_bank >= 0 && provider_bank <= TAGE_BANKS - 2) {
            for (int bi = provider_bank - 1; bi >= 0; bi--) {
                if (bank_match[bi]) { alt_bank = bi; break; }
            }
        } else if (provider_bank == TAGE_BANKS - 1) {
            for (int bi = TAGE_BANKS - 2; bi >= 0; bi--) {
                if (bank_match[bi]) { alt_bank = bi; break; }
            }
        }

        // If provider is weak, allow the alternate prediction to override
        if (provider_bank >= 0) {
            unsigned char pctr = tage_banks[provider_bank][bank_idx[provider_bank]].ctr;
            bool provider_weak = (pctr == 1 || pctr == 2);
            if (alt_bank >= 0 && provider_weak) {
                bool alt_pred = ctr_pred(tage_banks[alt_bank][bank_idx[alt_bank]].ctr);
                tage_guess = alt_pred;
            }
        }

        // ---- Global perceptron
        int prow = perc_row_idx(pc);
        int py   = perc_eval_row(prow, ghr);
        bool ppred = (py >= 0);

        // ---- Meta combiner: prefer TAGE or Local based on confidence
        int feats[META_FEATS];
        meta_build_feats(tage_guess, local_guess, ppred, feats, pc);
        int mrow = meta_row_idx(pc);
        int my   = meta_eval_row(mrow, feats);
        int amy  = (my >= 0) ? my : -my;

        // Chooser backs up the decision if meta confidence is low
        unsigned int cho_idx = get_chooser_idx(pc);
        bool prefer_tage = (amy >= META_TH) ? (my >= 0) : (chooser[cho_idx] >= 2);

        // Final decision between experts (unconditional → always taken)
        bool final_guess = conditional ? (prefer_tage ? tage_guess : local_guess) : true;

        // ---- Tie-breaking with perceptron when experts disagree and perceptron is confident
        int ay = (py >= 0) ? py : -py;
        if (conditional) {
            bool conflict = (tage_guess != local_guess);
            if (conflict && ay >= PERC_TH_TIE) {
                bool chose_tage = prefer_tage;
                bool ml_sides_loser =
                    (chose_tage  && (ppred == local_guess)) ||
                    (!chose_tage && (ppred == tage_guess));

                // Avoid overriding a strong TAGE provider
                bool provider_weak_ok = true;
                if (chose_tage) {
                    bool has_provider = (provider_bank >= 0);
                    if (has_provider) {
                        unsigned char pctr = tage_banks[provider_bank][bank_idx[provider_bank]].ctr;
                        provider_weak_ok = (pctr == 1 || pctr == 2);
                    }
                }
                if (ml_sides_loser && provider_weak_ok && (ppred != final_guess)) {
                    final_guess = ppred;
                }
            }
        }

        // ---- Loop predictor snapshot (not used to override here)
        bool loop_hit=false, loop_pred=false; unsigned char loop_conf=0;
        loop_predict(pc, conditional, loop_hit, loop_pred, loop_conf);

        // ---- Side-channel accumulators (sum three hashed slots)
        int sc_sum = 0;
        sc_sum += sc1[sc_idx1(pc)];
        sc_sum += sc2[sc_idx2(pc)];
        sc_sum += sc3[sc_idx3(pc)];

        // ---- Save all state for update()
        u.is_conditional  = conditional;
        u.final_pred      = final_guess;

        u.local_hist_idx  = lhist_idx;
        u.local_pht_idx   = lpht_idx;
        u.local_pred      = local_guess;

        u.chooser_idx     = cho_idx;

        u.tage_pred       = tage_guess;
        u.base_pred       = base_guess;
        u.provider_bank   = provider_bank;
        u.alt_bank        = alt_bank;
        u.base_idx        = bidx;
        for (int bi = 0; bi < TAGE_BANKS; bi++) {
            u.bank_idx[bi]   = bank_idx[bi];
            u.bank_match[bi] = bank_match[bi];
        }

        u.loop_hit = loop_hit; u.loop_pred = loop_pred; u.loop_conf = loop_conf;
        u.sc_sum   = sc_sum;

        u.perc_row = prow; u.perc_y = py; u.perc_pred = ppred;
        u.meta_row = mrow; u.meta_y  = my; u.meta_choose_tage = (my >= 0);

        // Report final prediction to simulator interface
        if (conditional) u.direction_prediction(final_guess);
        else             u.direction_prediction(true);
        u.target_prediction(0);
        return &u;
    }

    /*------------------------------------------------------------
      allocate_tage_entries()
      Replacement guided by 'u' (usefulness) counters:
        - Try shorter-history banks first ([TAGE_BANKS-2..0]) where no match.
        - Prefer entries with u==0; otherwise decay u to make room.
        - Special case: if both base & local are wrong and no provider,
          attempt allocation in the longest bank.
    ------------------------------------------------------------*/
    void allocate_tage_entries(my_update *mu, bool taken, unsigned int pc) {
        bool allocated = false;

        // Try banks [TAGE_BANKS-2 .. 0] where there was no tag match
        for (int bi = TAGE_BANKS - 2; bi >= 0; bi--) {
            if (mu->bank_match[bi]) continue;
            unsigned int idx = mu->bank_idx[bi];
            tage_entry_t &e = tage_banks[bi][idx];
            if (e.u == 0) {
                // Install fresh entry with strong initial opinion
                e.tag = tage_tag(bi, pc);
                e.ctr = taken ? 3 : 0;
                e.u   = 0;
                allocated = true;
                return;
            } else {
                // Age the entry to become replaceable
                if (e.u > 0) e.u--;
            }
        }

        // If nothing allocated and no provider, consider the longest bank
        if (!allocated && !mu->bank_match[TAGE_BANKS - 1] && mu->provider_bank < 0) {
            bool base_wrong  = (mu->base_pred  != taken);
            bool local_wrong = (mu->local_pred != taken);
            if (base_wrong && local_wrong) {
                unsigned int idx = mu->bank_idx[TAGE_BANKS - 1];
                tage_entry_t &e = tage_banks[TAGE_BANKS - 1][idx];
                if (e.u == 0) {
                    e.tag = tage_tag(TAGE_BANKS - 1, pc);
                    e.ctr = taken ? 3 : 0;
                    e.u   = 0;
                    allocated = true;
                } else {
                    if (e.u > 0) e.u--;
                }
            }
        }
    }

    /*------------------------------------------------------------
      update()
      Trains all components with the actual outcome:
        - Local: update PHT and per-PC local history shift.
        - Base:  2-bit counter update.
        - TAGE:  strengthen provider on hit; decay usefulness on miss,
                 and allocate new entries if needed.
        - Chooser: nudge toward the expert that was correct (TAGE vs Local).
        - Loop:   refine period, count, and confidence.
        - Side-channels: small saturating sums toward the outcome.
        - Perceptron: train only when TAGE and Local disagree.
        - Meta:   favor the expert that was right; weak-train near margin.
        - GHR:    shift in the latest outcome bit at the end.
    ------------------------------------------------------------*/
    void update (branch_update *u_base, bool taken, unsigned int target) {
        my_update *mu = (my_update*)u_base;

        if (mu->is_conditional) {
            // ---- Local predictor train
            ctr_update(local_pht[mu->local_pht_idx], taken);
            {
                // Shift in new local outcome bit (bounded width)
                unsigned short old_lh = local_history_table[mu->local_hist_idx];
                unsigned short new_lh = (unsigned short)((old_lh << 1) | (taken ? 1 : 0));
                unsigned short mask   = (1u << LOCAL_HISTORY_BITS) - 1u;
                local_history_table[mu->local_hist_idx] = (new_lh & mask);
            }

            // ---- Base bimodal train
            ctr_update(base_table[mu->base_idx], taken);

            // ---- TAGE train
            if (mu->provider_bank >= 0) {
                tage_entry_t &pe = tage_banks[mu->provider_bank][mu->bank_idx[mu->provider_bank]];
                ctr_update(pe.ctr, taken);
                bool tage_correct = (mu->tage_pred == taken);
                if (tage_correct) { if (pe.u < 3) pe.u++; }
                else              { if (pe.u > 0) pe.u--; allocate_tage_entries(mu, taken, last_bi.address); }
            } else {
                // No provider: allocate if base was wrong
                bool base_ok = (mu->base_pred == taken);
                if (!base_ok) allocate_tage_entries(mu, taken, last_bi.address);
            }

            // ---- Chooser train (prefer the expert that was right)
            bool tage_ok  = (mu->tage_pred  == taken);
            bool local_ok = (mu->local_pred == taken);
            if (tage_ok && !local_ok)      { if (chooser[mu->chooser_idx] < 3) chooser[mu->chooser_idx]++; }
            else if (local_ok && !tage_ok) { if (chooser[mu->chooser_idx] > 0) chooser[mu->chooser_idx]--; }

            // ---- Loop predictor online update
            loop_update(last_bi.address, taken);

            // ---- Side-channels: small saturating increments toward outcome
            auto sc_upd = [&](int8_t &x){
                if (taken) { if (x < 3) x++; }
                else       { if (x > -3) x--; }
            };
            sc_upd(sc1[sc_idx1(last_bi.address)]);
            sc_upd(sc2[sc_idx2(last_bi.address)]);
            sc_upd(sc3[sc_idx3(last_bi.address)]);

            // ---- Perceptron: only train when experts disagree
            bool conflict = (mu->tage_pred != mu->local_pred);
            if (conflict) {
                perc_train_row(mu->perc_row, taken, mu->perc_y, ghr);
            }

            // ---- Meta: prefer the expert that was right; otherwise weak-train near margin
            {
                int feats[META_FEATS];
                meta_build_feats(mu->tage_pred, mu->local_pred, mu->perc_pred, feats, last_bi.address);
                int label = 0;
                if (tage_ok && !local_ok) label = +1;       // prefer TAGE
                else if (!tage_ok && local_ok) label = -1;  // prefer Local

                if (label != 0) {
                    meta_train_row(mu->meta_row, feats, label);
                } else {
                    int y = mu->meta_y;
                    if (y < META_TH_TRAIN && y > -META_TH_TRAIN) {
                        // Use chooser tendency as pseudo-label near margin
                        int pseudo = (chooser[mu->chooser_idx] >= 2) ? +1 : -1;
                        meta_train_row(mu->meta_row, feats, pseudo);
                    }
                }
            }
        }

        // ---- Global history shift-in (only for conditional branches)
        if (mu->is_conditional) {
            ghr = ((ghr << 1) | (taken ? 1ull : 0ull));
            if (MAX_GHIST_LEN < 64) {
                unsigned long long mask = (1ull << MAX_GHIST_LEN) - 1ull;
                ghr &= mask;
            }
        }
        (void)target; // target not used in this predictor
    }
};
