"""
Analyze per-second cell count data
"""
import csv
import sys
from pathlib import Path


def analyze_per_second(csv_file):
    """Analyze per-second cell count CSV."""
    
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        return
    
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'second': int(row['second']),
                'avg_cell_count': float(row['avg_cell_count']),
                'min_cell_count': int(row['min_cell_count']),
                'max_cell_count': int(row['max_cell_count']),
                'frame_count': int(row['frame_count'])
            })
    
    if not data:
        print("No data found in CSV")
        return
    
    print("=" * 80)
    print("PER-SECOND CELL COUNT ANALYSIS")
    print("=" * 80)
    print(f"CSV File: {csv_file}")
    print(f"Total Duration: {data[-1]['second']} seconds")
    print()
    
    # Overall statistics
    avg_counts = [d['avg_cell_count'] for d in data]
    print("📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total Seconds Analyzed:   {len(data)}")
    print(f"Minimum Avg Cell Count:   {min(avg_counts):.2f} (Second {data[avg_counts.index(min(avg_counts))]['second']})")
    print(f"Maximum Avg Cell Count:   {max(avg_counts):.2f} (Second {data[avg_counts.index(max(avg_counts))]['second']})")
    print(f"Overall Average:          {sum(avg_counts)/len(avg_counts):.2f}")
    print()
    
    # Show each second's data
    print("📈 PER-SECOND BREAKDOWN")
    print("-" * 80)
    print(f"{'Second':>6} | {'Avg Cells':>10} | {'Min':>5} | {'Max':>5} | {'Frames':>6}")
    print("-" * 80)
    
    for d in data:
        print(f"{d['second']:>6} | {d['avg_cell_count']:>10.2f} | "
              f"{d['min_cell_count']:>5} | {d['max_cell_count']:>5} | "
              f"{d['frame_count']:>6}")
    
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_per_second.py <per_second_csv_file>")
        sys.exit(1)
    
    analyze_per_second(sys.argv[1])
