# System Architecture

## Overview

The Crop-Water-Soil Interaction Modeling Platform uses a **multi-task learning (MTL)** architecture to jointly model three interconnected agricultural systems.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT SOURCES                                 │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│  Satellite  │   Weather   │    Soil     │        Water            │
│  (NDVI/EVI) │  (Temp,Rain)│  (NPK, OC)  │  (GW level, ET)         │
└──────┬──────┴──────┬──────┴──────┬──────┴────────────┬────────────┘
       │             │             │                   │
       └─────────────┴─────────────┴───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Feature Alignment   │
                    │   & Normalization     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   SHARED ENCODER      │
                    │   (LSTM/CNN/Transf.)  │
                    │   - Temporal patterns │
                    │   - Cross-modal rep.  │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  CROP HEAD    │   │  SOIL HEAD    │   │  WATER HEAD   │
    │  (NDVI/Yield) │   │  (NPK, OC)    │   │  (GW Stress)  │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  CONSTRAINT LAYER     │
                    │  - Physical validation│
                    │  - Penalty computation│
                    └───────────────────────┘
```

## Domain Constraints

1. **Groundwater-Rainfall Coupling**: If rainfall ↓ and irrigation ↑ → groundwater should not increase
2. **Soil-Crop Coupling**: Poor soil nutrients → crop health cannot suddenly spike
3. **ET Limits**: Crop water demand must obey evapotranspiration bounds

## Implementation Notes

- Shared encoder captures cross-task patterns (e.g., drought affects all three)
- Task-specific heads allow specialized prediction for each domain
- Constraint penalty is added to the multi-task loss during training
