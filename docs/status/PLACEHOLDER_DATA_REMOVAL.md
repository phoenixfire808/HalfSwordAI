# Placeholder Data Removal - Implementation Summary

## Overview
All placeholder/mock data has been removed and replaced with real data sources using visual detection and OCR.

## Changes Made

### 1. MemoryReader (`half_sword_ai/perception/vision.py`)

**Before:**
- Used placeholder memory offsets (0x12345678, etc.)
- Returned mock state with fake values (health: 100.0, stamina: 100.0, etc.)
- Position always returned (0.0, 0.0, 0.0)

**After:**
- Memory offsets set to `None` (not fake addresses)
- Returns `_get_visual_state()` with `None` values when data unavailable
- Position returns `None` values instead of fake zeros
- All values explicitly marked as "unknown" rather than fake defaults
- Added `data_source` field to indicate where data came from

**Key Changes:**
```python
# OLD: Returned fake 100.0 health
return {"health": 100.0, "stamina": 100.0, ...}

# NEW: Returns None when unknown
return {
    "health": None,  # Unknown - not fake 100
    "stamina": None,  # Unknown - not fake 100
    "data_source": "visual",
    "note": "Using visual detection - some values may be unknown"
}
```

### 2. Actor Process (`half_sword_ai/core/actor.py`)

**Added Real Data Sources:**
- **OCR Reward Tracker**: Real score tracking from Abyss mode
- **Terminal State Detector**: Real death screen detection
- Integrated both into game state and reward calculation

**Reward Calculation:**
- **Before**: Used default values (health: 100.0) even when unknown
- **After**: 
  - Uses OCR for real score rewards
  - Uses terminal state detector for real death detection
  - Only uses memory data if `data_source == 'memory'` and values are not defaults
  - Avoids using mock values (100.0, 0.0, etc.) when data is actually unknown

**Game State Updates:**
- Terminal state detection integrated into main loop
- OCR score tracking integrated into main loop
- Real detection results override any placeholder values

### 3. Configuration

**New Settings:**
- `OCR_ENABLED`: Enable/disable OCR reward tracking
- `TERMINAL_STATE_DETECTION`: Enable/disable death screen detection

## Data Flow

### Real Data Sources (Priority Order):

1. **OCR Reward Tracker** (if enabled)
   - Reads actual score from Abyss mode UI
   - Provides real score deltas for rewards
   - Updates every 30 frames (optimized)

2. **Terminal State Detector** (if enabled)
   - Detects death screen using color histogram analysis
   - Provides real terminal state detection
   - No OCR overhead - fast detection

3. **Memory Reader** (if available)
   - Only used if `data_source == 'memory'`
   - Values checked to avoid default/mock values
   - Returns `None` when data unavailable

4. **Visual Detection** (fallback)
   - Returns `None` values when data unknown
   - No fake defaults

## Benefits

1. **No Fake Training Data**: Model trains on real observations only
2. **Accurate Rewards**: OCR provides real score-based rewards
3. **Real Terminal States**: Death detection uses actual visual analysis
4. **Transparent Unknowns**: System explicitly marks unknown values as `None`
5. **Better Learning**: Model learns from real game state, not placeholder values

## Testing

To verify no placeholder data:
1. Check logs for "Using placeholder" warnings - should be removed
2. Check game_state values - should be `None` when unknown, not fake defaults
3. Verify OCR rewards are being tracked (check logs for "OCR reward" messages)
4. Verify terminal state detection (check logs for "Terminal state detected")

## Migration Notes

- Old code that expected default values (100.0 health) may need updates
- Check for `None` values when reading game state
- Use `data_source` field to determine data reliability
- OCR and terminal detection are optional - system works without them but uses visual fallback

