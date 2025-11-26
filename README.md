# 📘 Hybrid Branch Predictor (TAGE + Local + ML Meta-Arbitration)

**Author:** Bingyu Chen
**Course:** UCLA ECE M116C – Computer Architecture (CA2)
**Project:** Custom Hybrid Branch Predictor for CBP2 Infrastructure
**Files:** `my_predictor.h` (implementation), CA2 PDF report (design & results)

This repository contains a high-accuracy hybrid branch predictor combining **TAGE**, **Local History**, **Perceptron ML**, **Meta-Chooser**, **Loop prediction**, and **Side-Channel bias units**.
The design is optimized for the CBP2 infrastructure and achieves competitive **MPKI performance** with modest storage (~1.65 MB implemented).

---

## 🚀 Overview

This predictor integrates multiple complementary prediction mechanisms:

* **5-Bank TAGE** (history lengths {4, 8, 16, 32, 64})
* **Local Predictor** (LHT14 + PHT 2-bit counters)
* **Base Predictor** (bimodal 2-bit)
* **Conflict-Only Perceptron** (64 history bits + bias)
* **19-Feature Meta-Perceptron** for arbitration
* **2-bit Chooser** fallback
* **Lightweight Loop Predictor**
* **Side-Channel hashed sums** (SC1/SC2/SC3)

The design philosophy is:

> *Let TAGE drive global correlations, let Local capture per-PC behavior, and let ML intervene only when necessary — ensuring stability and low noise.*
>

---

## 🧠 High-Level Architecture

### 🟦 1. Base Predictor

Simple bimodal 2-bit counter table (2¹⁶ entries).

### 🟧 2. Local Predictor

* Per-PC **14-bit local history table**
* **PHT** (2¹⁴ entries), 2-bit counters
* Captures idiosyncratic per-branch patterns.

### 🟩 3. TAGE

Five banks using different history lengths:

```
{ 4, 8, 16, 32, 64 } 
```

Each entry stores:

* 16-bit tag
* 2-bit counter (stored in uint8)
* 2-bit usefulness `u` (stored in uint8)


The longest bank only participates when:

* Counter is **strong** (0 or 3), and
* Usefulness `u ≥ 1`


### 🟥 4. Conflict-Only Perceptron

* 64 GH bits → ±1 features
* Trains only when **TAGE and Local disagree**
* Clips weights to int8 range


### 🟪 5. Meta-Chooser Perceptron (19 features)

Features:

* sign(TAGE), sign(Local), sign(Perceptron)
* 8 LSBs of GHR
* 8 LSBs of PC


If |meta_score| < threshold → use **2-bit chooser fallback**.

### 🟨 6. Loop Predictor

Tracks:

* Loop period
* Iteration count
* Confidence


### 🟫 7. Side-Channels (SC-lite)

Three tiny hashed accumulators (int8) add a small bias.

---

## 🔄 Prediction Flow

1. Read **Local**, **Base**, **TAGE** candidates
2. Identify **provider** and possibly an **alt bank**
3. Evaluate **global perceptron** (only used when experts conflict)
4. Build 19-D meta features → **meta-perceptron**
5. If meta confident → choose TAGE or Local
6. Else fallback to **2-bit chooser**
7. Perceptron may overturn if:

   * Experts conflict
   * Perceptron has high confidence
   * Provider is weak



---

## 📉 Training Flow

Upon receiving the actual outcome:

* Local PHT counter update
* Base counter update
* TAGE counter update + usefulness update
* Allocation when provider is wrong or absent
* Chooser update
* Perceptron update on conflicts
* Meta perceptron update
* Loop predictor refinement
* Side channel small updates
* GHR shift


---

## 📦 Storage Summary

| Component                | Size            |
| ------------------------ | --------------- |
| Base Table               | 524,288 bits    |
| 5× TAGE Banks            | 10,485,760 bits |
| Local Tables (LHT + PHT) | 1,179,648 bits  |
| Chooser Table            | 524,288 bits    |
| Loop Predictor           | 294,912 bits    |
| Side Channels            | 98,304 bits     |
| Perceptron               | 532,480 bits    |
| Meta Perceptron          | 163,840 bits    |

**Total implemented size: ~1.65 MB**


---

## 📊 Performance (MPKI)

Results on CBP2 benchmark traces:

| Trace     | MPKI       |
| --------- | ---------- |
| 164.gzip  | 10.357     |
| 175.vpr   | 9.585      |
| 176.gcc   | 4.475      |
| 181.mcf   | 11.835     |
| …         | …          |
| 256.bzip2 | **0.044**  |
| 300.twolf | **13.941** |

**Average MPKI: 3.829**


---

## ⚙️ Build & Run (CBP2 Infrastructure)

If you want, I can generate a full section with:

* Makefile
* Build commands
* Folder structure
* Sample CBP2 output/log
* Tips for debugging TAGE/Local/Meta interactions

Just say **“add build instructions”**.

---

## 🙌 Acknowledgements

This project was implemented for **ECE M116C (Computer Architecture)** at UCLA, taught by Prof.Nader Sehatbakhsh.
