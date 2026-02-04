# Fixture Differentiation Ideas (Brainstorm)

## Features to extract per event
- duration_s
- total_volume_L
- avg_flow / max_flow / min_flow
- flow_std (variability)
- peak_count (number of short high-flow spikes)
- rise_time / fall_time
- duty_cycle (fraction of time flow > threshold)

## Typical signatures (rough)
- Urinal: small volume, short duration, sharp burst
- Water closet (WC): medium volume, short burst, higher peak
- Basin: lower flow, short-to-medium duration, variable on/off
- Shower: moderate flow, long duration, steady profile

## Approaches
- Rule-based thresholds on volume/duration/peak flow
- Clustering on event features, then label clusters
- Supervised classifier (RF/XGBoost) with labeled events
