# Volt X Training Data Setup - Summary

## ✅ Completed Setup

### Directory Structure Created
All training data is now stored on **D: drive** (saving C: drive space):

```
D:\VoltData\
├── phase0/               # Bootstrap data
│   ├── humaneval/        # ✅ Downloaded (164 problems)
│   ├── mbpp/             # ✅ Downloaded (257 problems)
│   ├── exercism/         # Pending (manual download)
│   └── the_stack_sample/ # ✅ Downloaded (6,558,984 files, 43.83 GB)
├── phase1/               # Translator training
│   └── codesearchnet/    # ✅ Downloaded (100K pairs, 144 MB)
├── phase2/               # VFN training
│   ├── apps/             # ✅ Downloaded (10,000 problems, ~1.34 GB)
│   ├── clrs/             # Pending (generate locally)
│   └── mbpp_full/        # (uses phase0 data)
├── phase3/               # Hard Core calibration
│   ├── routing_labels/   # Pending (generate)
│   └── safety_examples/  # Pending (generate)
├── phase5/               # Scale & benchmark
│   ├── the_stack_multilang/ # Pending (120GB)
│   ├── multiple/         # ✅ Downloaded (2,388 problems)
│   └── ds1000/           # Pending
└── scripts/              # Download scripts (ready to use)
```

### Documentation Created
1. **D:\VoltData\DATA.md** - Full dataset documentation
2. **D:\VoltData\DOWNLOAD_STATUS.md** - Current download status
3. **D:\VoltData\scripts\README.md** - Script usage guide
4. **c:\Volt_X\training_config.toml** - Configuration file with dataset paths
5. **c:\Volt_X\CODE_TRAINING_PLAN.md** - Complete code training plan

### Download Scripts Ready
All scripts are in `D:\VoltData\scripts\`:
- ✅ `download_small_datasets.py` - HumanEval, MBPP, MultiPL-E
- ✅ `download_apps.py` - APPS competitive programming
- ✅ `download_codesearchnet.py` - Function-docstring pairs
- ✅ `download_the_stack.py` - Large code corpus (with sampling)

### Successfully Downloaded Datasets

| Dataset | Problems/Files | Size | Location |
|---------|----------------|------|----------|
| HumanEval | 164 | 0.20 MB | `D:\VoltData\phase0\humaneval\data\` |
| MBPP | 257 | 0.12 MB | `D:\VoltData\phase0\mbpp\data\` |
| The Stack (Python) | 6,558,984 files | 43.83 GB | `D:\VoltData\phase0\the_stack_sample\` |
| CodeSearchNet (Python) | 100,000 pairs | 144 MB | `D:\VoltData\phase1\codesearchnet\data\` |
| APPS | 10,000 | ~1.34 GB | `D:\VoltData\phase2\apps\data\` |
| MultiPL-E | 2,388 | 2.89 MB | `D:\VoltData\phase5\multiple\data\` |
| **Total** | **12,809 problems + 6.5M files** | **~45.3 GB** | |

MultiPL-E includes HumanEval translated to 15 languages:
- JavaScript, Java, C++, Go, C#, PHP, Ruby, Swift
- TypeScript, Scala, Racket, OCaml, Lua, R, Julia

## 📋 Next Steps

### Phase 0 (Bootstrap) - COMPLETE
- ✅ VFN Checkpoint (0.1)
- ✅ Code Dataset Pipeline (0.2)
- ✅ Codebook Init Pipeline (0.3) — code done, k-means deferred to after Phase 1
- ✅ Code Attention Bias (0.4)

### Phase 1 (Translator) - COMPLETE
- ✅ BPE tokenizer trained (32K vocab from The Stack)
- ✅ CNN encoder trained (contrastive loss 1.41, valid 1.69)
- ✅ Autoregressive decoder trained (61% token accuracy)
- ✅ Role labeling heuristics implemented

### Phase 2 (VFN Training) - Next
- ✅ APPS dataset downloaded (10K problems)
- ⏳ Need: CLRS algorithm traces (generate locally)
- **Action:** Scale VFN from 525K → 50M params, train on code tasks

### Phase 3-5 - Later
Generate synthetic datasets as needed during those phases.

## 🎯 Recommended Download Order

**Today (Quick - 30 min total):**
```bash
cd D:\VoltData\scripts

# Already done:
# - HumanEval ✅
# - MBPP ✅
# - MultiPL-E ✅

# Download medium datasets:
python download_apps.py                           # ~1 GB, 15-30 min
python download_codesearchnet.py --sample 100000  # ~500 MB, 10-20 min
```

**Overnight (The Stack - 1-3 hours):**
```bash
cd D:\VoltData\scripts
python download_the_stack.py --language python --size 50 --phase 0
```

**Later (when ready for Phase 5):**
```bash
cd D:\VoltData\scripts
python download_the_stack.py --multi-language --size 120 --phase 5  # 3-8 hours
```

## 🔧 Configuration

### Dataset Paths in Code
All training code should reference datasets via `c:\Volt_X\training_config.toml`:

```toml
[paths]
data_root = "D:\\VoltData"
humaneval_path = "D:\\VoltData\\phase0\\humaneval\\data"
mbpp_path = "D:\\VoltData\\phase0\\mbpp\\data"
# ... etc
```

**Never hardcode C: drive paths for datasets!**

### Example Usage in Rust
```rust
use config::Config;

let settings = Config::builder()
    .add_source(config::File::with_name("training_config"))
    .build()?;

let humaneval_path = settings.get_string("paths.humaneval_path")?;
// Load dataset from D: drive
```

### Example Usage in Python
```python
import toml

config = toml.load("c:/Volt_X/training_config.toml")
humaneval_path = config["paths"]["humaneval_path"]
# Load dataset from D: drive
```

## 📊 Disk Space Status

### Current Usage
- Downloaded: ~45.3 GB (The Stack 43.8GB, APPS 1.3GB, CodeSearchNet 144MB, rest ~3MB)
- Checkpoints: ~20 MB (encoder + decoder + tokenizer)
- Scripts: ~50 KB

### Projected Usage (Full Training)
- Phase 0-1: ~45.3 GB (DONE)
- Phase 2: ~2 GB
- Phase 3: ~100 MB
- Phase 5: ~120 GB
- **Total:** ~167 GB

**Ensure D: drive has 200+ GB free before large downloads.**

## ✅ Integration Complete

The training data setup is now integrated into the project:
- `CLAUDE.md` updated with data location info
- `training_config.toml` created with all paths
- All future AI assistants will know datasets are on D: drive
- Memory system (`C:\Users\where\.claude\projects\c--Volt-X\memory\`) updated

## Training Progress

Phase 0 (Bootstrap) and Phase 1 (Translator) are **complete**.

- Phase 0.1 (VFN Checkpoint): DONE
- Phase 0.2 (Code Dataset Pipeline): DONE
- Phase 0.3 (Codebook Init Pipeline): Code DONE, k-means deferred
- Phase 0.4 (Code Attention Bias): DONE
- Phase 1.1 (Code Encoder): DONE — Contrastive loss 1.41
- Phase 1.2 (Code Decoder): DONE — 61% token accuracy
- Phase 1.3 (Role Grounding): DONE — Joint with encoder

**Next: Phase 2 — VFN Flow Matching on Code Tasks**
