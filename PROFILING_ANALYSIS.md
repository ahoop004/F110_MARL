# F110_MARL Performance Profiling Analysis

## Executive Summary

Profiling reveals **FTG LiDAR preprocessing is the primary bottleneck**, consuming **97% of FTG agent action selection time**.

### Key Findings

**FTG Agent Action Selection** (5000 calls in 81.6s):
- **Throughput**: 61.3 calls/sec
- **Average latency**: 16.3ms per call
- **Bottleneck**: `preprocess_lidar()` - 79.3s (97% of total time)

---

## Detailed Analysis

### Hot Path: FTG `preprocess_lidar()`

```
Total time: 81.621s for 5000 calls
└─ preprocess_lidar: 79.339s (97.2%)
   ├─ np.mean() calls: 43.714s (53.5%)
   ├─ np.clip() calls: 20.962s (25.7%)
   └─ Other: 14.663s (18.0%)
```

**Problem**: The method processes 1080-beam LiDAR by:
1. **For each of 1080 beams** (outer loop):
   - Slice array window (start:end)
   - Call `np.clip()` on slice
   - Call `np.mean()` on clipped slice
   - Append result to array

This results in:
- **5,410,000 `np.mean()` calls** (1082 per LiDAR scan)
- **5,440,000 `np.clip()` calls** (1088 per LiDAR scan)
- Massive function call overhead
- Poor cache utilization (many small array operations)

### Current Implementation

```python
def preprocess_lidar(self, ranges, min_scan: Optional[float] = None) -> np.ndarray:
    N = len(ranges)
    window = self._adaptive_window_size(min_scan)
    half = window // 2
    proc = np.empty(N, dtype=np.float32)  # ← Already optimized in Phase 5

    for i in range(N):  # ← Main bottleneck: 1080 iterations
        start = max(0, i - half)
        end = min(N - 1, i + half)
        avg = np.mean(np.clip(ranges[start:end+1], 0, self.max_distance))  # ← Hot line
        proc[i] = avg

    return proc
```

---

## Optimization Opportunities

### 1. **Vectorized Moving Average** (High Impact)

**Replace loop with vectorized operations using convolution or stride tricks**

```python
def preprocess_lidar_vectorized(self, ranges, min_scan: Optional[float] = None) -> np.ndarray:
    """Vectorized LiDAR preprocessing using numpy convolution."""
    N = len(ranges)
    window = self._adaptive_window_size(min_scan)

    # Clip entire array once
    clipped = np.clip(ranges, 0, self.max_distance)

    # Use np.convolve for moving average (much faster than loop)
    kernel = np.ones(window, dtype=np.float32) / window

    # 'same' mode keeps output same size as input
    proc = np.convolve(clipped, kernel, mode='same')

    return proc.astype(np.float32)
```

**Expected speedup**: **10-20x** (from 16ms to 0.8-1.6ms per call)

**Benefits**:
- Single `np.clip()` call instead of 1088
- Single `np.convolve()` call instead of 1082 `np.mean()` calls
- Better cache utilization
- SIMD vectorization

**Trade-off**:
- Edge handling differs slightly (convolution wraps vs current max/min clamping)
- Can be addressed with padding if needed

---

### 2. **Numba JIT Compilation** (Medium Impact)

**JIT-compile the loop for near-C performance**

```python
from numba import njit

@njit(cache=True, fastmath=True)
def _preprocess_lidar_jit(
    ranges: np.ndarray,
    window: int,
    max_distance: float,
) -> np.ndarray:
    N = len(ranges)
    half = window // 2
    proc = np.empty(N, dtype=np.float32)

    for i in range(N):
        start = max(0, i - half)
        end = min(N, i + half + 1)

        # Manual mean computation (faster in JIT)
        total = 0.0
        count = 0
        for j in range(start, end):
            val = ranges[j]
            if val < 0:
                val = 0.0
            elif val > max_distance:
                val = max_distance
            total += val
            count += 1

        proc[i] = total / count if count > 0 else 0.0

    return proc

def preprocess_lidar(self, ranges, min_scan: Optional[float] = None) -> np.ndarray:
    window = self._adaptive_window_size(min_scan)
    return _preprocess_lidar_jit(
        np.asarray(ranges, dtype=np.float32),
        window,
        self.max_distance
    )
```

**Expected speedup**: **5-8x** (from 16ms to 2-3ms per call)

**Benefits**:
- Eliminates Python function call overhead
- Eliminates numpy array creation overhead
- Better loop optimization

---

### 3. **Hybrid Approach** (Best of Both)

Use vectorized approach for typical cases, JIT for edge cases:

```python
def preprocess_lidar(self, ranges, min_scan: Optional[float] = None) -> np.ndarray:
    window = self._adaptive_window_size(min_scan)

    # Fast path: use vectorized convolution
    if window <= 10:  # Most common case
        return self._preprocess_vectorized(ranges, window)
    # Slow path: use JIT for large windows
    else:
        return self._preprocess_jit(ranges, window)
```

---

## Other Observations

### `create_bubble()` - Already Efficient ✓

```
Time: ~0.2s out of 81.6s (0.2%)
```

In-place modification is efficient. No optimization needed.

### `_compute_speed()` - Minor Bottleneck

```
Time: 0.649s out of 81.6s (0.8%)
```

Very minor compared to preprocessing. Not worth optimizing now.

---

## Recommendations

### Immediate (High ROI)

1. **✅ Implement vectorized moving average**
   - **Effort**: Low (20 lines of code)
   - **Impact**: 10-20x speedup on FTG hot path
   - **Risk**: Low (easy to test)

### Short-term (Medium ROI)

2. **Consider Numba JIT** if vectorization has edge case issues
   - **Effort**: Medium (need to handle edge cases carefully)
   - **Impact**: 5-8x speedup
   - **Risk**: Medium (Numba adds dependency complexity)

### Long-term

3. **Profile full training loop** once FTG is optimized
   - Identify next bottlenecks in environment stepping
   - Profile RL algorithm update loops

---

## Expected Impact

### Current Performance
- FTG action selection: **61.3 calls/sec** (16.3ms each)
- Dominated by LiDAR preprocessing: **15.9ms** (97%)

### After Vectorization
- FTG action selection: **~600-1200 calls/sec** (0.8-1.6ms each)
- **10-20x speedup** in FTG agents
- **Training throughput increase**: ~10x for FTG-heavy scenarios

### Combined with Previous Optimizations

**Total speedup from all phases**:
- Phase 1-3: ~3-4x (training loop, algorithms, environment)
- Phase 4: Robustness (no speed change)
- Phase 5: ~2-3x (memory, array pre-allocation, FTG preprocessing)
- **Phase 6 (proposed)**: ~10-20x (vectorized LiDAR)

**Overall**: **~30-80x total speedup** from baseline! 🚀

---

## Next Steps

1. Implement vectorized `preprocess_lidar()`
2. Benchmark against current implementation
3. Verify correctness with unit tests
4. Profile full training loop to find next bottleneck
5. Consider GPU acceleration for RL updates if CPU-bound
