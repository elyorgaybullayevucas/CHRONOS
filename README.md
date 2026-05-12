# CHRONOS — TKG Extrapolation Model

**Cross-scale Historical Reasoning and Ontological Network for Ordered Sequence-based link prediction**

## Arxitektura

```
INPUT (s, r, t, paths, history)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER                                                       │
│  ent_emb[s] + RPE(r,t) + TFE(r, Δt)                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ Entity Embedding│  │ RPE             │  │ TFE ⭐ (YANGI)        │  │
│  │ (E × ent_dim)   │  │ Relation Profile│  │ Per-relation         │  │
│  │                 │  │ Encoding        │  │ frequency features   │  │
│  └─────────────────┘  └─────────────────┘  └──────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
        │                                │
        ▼                                ▼
┌────────────────────┐      ┌────────────────────────────────────────────┐
│  PATH SIGNAL       │      │  HISTORY SIGNAL                            │
│                    │      │                                            │
│  PathEncoder       │      │  HistoryEncoder (EI-HT self-attn)          │
│  (EI-HT: entity-   │      │        │                                   │
│   independent)     │      │        ▼                                   │
│        │           │      │  QATS ⭐ (YANGI)                           │
│        ▼           │      │  Query-Aware Temporal Selection            │
│  TPL (K=128        │      │  Cross-attention: query × history          │
│  learned patterns) │      │        │                                   │
│        │           │      │        ▼                                   │
│        ▼           │      │  TRTM ⭐ (YANGI)                           │
│  path_signal (B,D) │      │  Temporal Relation Transition Memory       │
│  + PCL loss ⭐     │      │  (r_prev, r_curr, Δt_bin) → transition     │
└────────────────────┘      └────────────────────────────────────────────┘
        │                                │
        └─────────────┬──────────────────┘
                      ▼
             ┌─────────────────┐
             │  3SF FUSION     │
             │  entity +       │
             │  path +         │
             │  history        │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │  Scoring Head   │
             │  query_proj ·   │
             │  ent_emb[all]   │
             └────────┬────────┘
                      ▼
             ┌─────────────────────────────┐
             │  LOSSES                     │
             │  L = w_link · L_link        │
             │    + w_adv  · L_self_adv    │
             │    + w_pcl  · L_PCL  ⭐     │
             │    + w_orth · L_ortho       │
             └─────────────────────────────┘
```

## Yangi komponentlar

### 1. TFE — Temporal Frequency Encoding
Har bir relatsiya uchun vaqt chastotasi xususiyatlarini o'rganadi.
- `sin(freq_r × Δt)` + `cos(freq_r × Δt)` — K ta chastota
- Kunlik/haftalik/yillik naqshlarni capture qiladi

### 2. QATS — Query-Aware Temporal Selection
Cross-attention bilan tarix elementlarini so'rov ga mos ravishda tanlaydi.
- EI-HT faqat tarix ichida self-attention qiladi
- QATS so'rov vektorini query sifatida ishlatib eng relevant faktlarni tanlaydi
- Gated residual: qancha history signali qo'shilishini nazorat qiladi

### 3. TRTM — Temporal Relation Transition Memory
Relatsiya o'tish naqshlarini Δt bilan birga modellaydi.
- `r_src ⊗ bin_emb[Δt] ⊗ r_dst` → trilinear transition scorer
- "Kelishuv → harbiy harakat" kabi temporal causal naqshlarni o'rganadi

### 4. PCL — Path Contrastive Learning (InfoNCE)
Yo'l representatsiyalarini regularizatsiya qiladi.
- `L_PCL = -log(exp(q·p+ / τ) / Σ exp(q·p- / τ))`
- Har bir query o'z yo'llariga yaqin, boshqa querylarning yo'llaridan uzoq bo'ladi

## Ishlatish

```bash
# Ma'lumotlarni bog'lash (tkg_elite2 dan)
mklink /D data ..\tkg_elite2\data    # Windows symlink
# yoki
cp -r ../tkg_elite2/data ./data      # Nusxalash

# O'qitish
python main.py --dataset ICEWS18
python main.py --dataset WIKI
python main.py --dataset YAGO --epochs 500
python main.py --dataset GDELT

# Resume
python main.py --dataset ICEWS18 --resume checkpoints/ICEWS18_best.pt
```

## Talablar

```
torch >= 1.13
numpy
```

## Fayl strukturasi

```
chronos/
├── config.py              # Barcha hyperparametrlar
├── main.py                # Kirish nuqtasi
├── models/
│   └── chronos_model.py   # CHRONOS arxitekturasi (barcha komponentlar)
├── data/
│   ├── dataset.py         # CHRONOSDataset
│   └── datamodule.py      # DataLoader, collate_fn
├── trainers/
│   └── trainer.py         # O'qitish/baholash sikli
└── utils/
    ├── paths.py            # Temporal path sampling
    ├── metrics.py          # MRR, Hits@K
    └── logging.py          # Logger
```
