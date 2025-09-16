import numpy as np

def preprocess_lidar(ranges, max_distance=30.0, window_size=5):
    processed = []
    N = len(ranges)
    half_window = window_size // 2
    for i in range(N):
        start_idx = max(0, i - half_window)
        end_idx = min(N - 1, i + half_window)
        avg = np.mean(np.clip(ranges[start_idx:end_idx+1], 0, max_distance))
        processed.append(avg)
    return np.array(processed)

def create_bubble(processed_lidar, bubble_radius=30):
    closest_point = np.argmin(processed_lidar)
    start_idx = max(closest_point - bubble_radius, 0)
    end_idx = min(closest_point + bubble_radius, len(processed_lidar) - 1)
    processed_lidar[start_idx:end_idx+1] = 0
    return processed_lidar

def find_max_gap(processed_lidar, threshold=0.5):
    masked = processed_lidar > threshold
    gaps = []
    gap_start = None
    for i, val in enumerate(masked):
        if val and gap_start is None:
            gap_start = i
        elif not val and gap_start is not None:
            gaps.append((gap_start, i - 1))
            gap_start = None
    if gap_start is not None:
        gaps.append((gap_start, len(masked) - 1))

    if not gaps:
        return 0, len(masked) - 1  # Default to entire scan if no gaps found
    
    max_gap = max(gaps, key=lambda x: x[1] - x[0])
    return max_gap

def find_best_point(gap):
    return (gap[0] + gap[1]) // 2

def gap_follow_action(scan_ranges, angle_min=-4.7/2, angle_increment=4.7/1080):
    proc_lidar = preprocess_lidar(scan_ranges)
    proc_lidar = create_bubble(proc_lidar)
    gap = find_max_gap(proc_lidar)
    best_point = find_best_point(gap)

    steering_angle = angle_min + best_point * angle_increment

    if abs(steering_angle) < np.radians(10):
        speed = 4.0
    elif abs(steering_angle) < np.radians(20):
        speed = 3.5
    else:
        speed = 2.5

    return np.array([steering_angle, speed])
