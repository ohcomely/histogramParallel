#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def load_data(benchmark_file, histogram_file):
    """Load benchmark and histogram data from CSV files."""
    benchmark_df = pd.read_csv(benchmark_file)
    histogram_df = pd.read_csv(histogram_file)
    return benchmark_df, histogram_df

def plot_performance_comparison(benchmark_df, output_dir):
    """Create bar plot comparing performance of different implementations."""
    plt.figure(figsize=(12, 6))

    # Get single-core performance as baseline
    baseline = benchmark_df[benchmark_df['Implementation'] == 'CPU-SingleCore']['Time(ms)'].iloc[0]

    # Prepare data for plotting
    implementations = []
    times = []
    speedups = []

    # Add single-core result
    implementations.append('CPU\n(1 thread)')
    times.append(benchmark_df[benchmark_df['Implementation'] == 'CPU-SingleCore']['Time(ms)'].iloc[0])
    speedups.append(1.0)

    # Add best multi-core result
    multicore_df = benchmark_df[benchmark_df['Implementation'] == 'CPU-MultiCore']
    best_multicore = multicore_df.loc[multicore_df['Time(ms)'].idxmin()]
    implementations.append(f'CPU\n({int(best_multicore["Threads"])} threads)')
    times.append(best_multicore['Time(ms)'])
    speedups.append(baseline / best_multicore['Time(ms)'])

    # Add GPU result
    gpu_time = benchmark_df[benchmark_df['Implementation'] == 'GPU']['Time(ms)'].iloc[0]
    implementations.append('GPU')
    times.append(gpu_time)
    speedups.append(baseline / gpu_time)

    # Create subplot for execution times
    plt.subplot(1, 2, 1)
    bars = plt.bar(implementations, times)
    plt.title('Execution Time Comparison')
    plt.ylabel('Time (ms)')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}ms',
                 ha='center', va='bottom')

    # Create subplot for speedups
    plt.subplot(1, 2, 2)
    bars = plt.bar(implementations, speedups)
    plt.title('Speedup vs Single-Core CPU')
    plt.ylabel('Speedup Factor')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}x',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cpu_scaling(benchmark_df, output_dir):
    """Create line plot showing CPU performance scaling with thread count."""
    multicore_df = benchmark_df[benchmark_df['Implementation'] == 'CPU-MultiCore'].copy()
    baseline_time = benchmark_df[benchmark_df['Implementation'] == 'CPU-SingleCore']['Time(ms)'].iloc[0]
    multicore_df['Speedup'] = baseline_time / multicore_df['Time(ms)']

    plt.figure(figsize=(10, 6))

    # Plot actual speedup
    plt.plot(multicore_df['Threads'], multicore_df['Speedup'],
             marker='o', linewidth=2, label='Actual Speedup')

    # Plot ideal linear speedup
    max_threads = multicore_df['Threads'].max()
    ideal_line = np.linspace(1, max_threads, 100)
    plt.plot(ideal_line, ideal_line, '--', label='Linear Speedup', alpha=0.7)

    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup vs Single-Thread')
    plt.title('CPU Performance Scaling')
    plt.grid(True)
    plt.legend()

    # Set x-axis to show actual thread counts
    plt.xticks(multicore_df['Threads'])

    plt.savefig(os.path.join(output_dir, 'cpu_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_throughput_comparison(benchmark_df, output_dir):
    """Create bar plot comparing throughput of different implementations."""
    plt.figure(figsize=(10, 6))

    implementations = []
    throughputs = []

    # Add single-core result
    implementations.append('CPU\n(1 thread)')
    throughputs.append(benchmark_df[benchmark_df['Implementation'] == 'CPU-SingleCore']['Throughput(MP/s)'].iloc[0])

    # Add best multi-core result
    multicore_df = benchmark_df[benchmark_df['Implementation'] == 'CPU-MultiCore']
    best_multicore = multicore_df.loc[multicore_df['Throughput(MP/s)'].idxmax()]
    implementations.append(f'CPU\n({int(best_multicore["Threads"])} threads)')
    throughputs.append(best_multicore['Throughput(MP/s)'])

    # Add GPU result
    implementations.append('GPU')
    throughputs.append(benchmark_df[benchmark_df['Implementation'] == 'GPU']['Throughput(MP/s)'].iloc[0])

    bars = plt.bar(implementations, throughputs)
    plt.title('Processing Throughput Comparison')
    plt.ylabel('Throughput (Million Pixels/sec)')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f} MP/s',
                 ha='center', va='bottom')

    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_histogram_distribution(histogram_df, output_dir):
    """Create line plots showing RGB channel distributions."""
    plt.figure(figsize=(12, 6))

    channels = ['Red', 'Green', 'Blue']
    for channel in channels:
        plt.plot(histogram_df['Bin'], histogram_df[channel],
                 label=channel, alpha=0.7)

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('RGB Channel Distributions')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, 'rgb_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'build/bin/Images')
    output_dir = os.path.join(data_dir, 'plots')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    benchmark_file = os.path.join(data_dir, 'histogram_benchmark.csv')
    histogram_file = os.path.join(data_dir, 'histogram_data.csv')

    benchmark_df, histogram_df = load_data(benchmark_file, histogram_file)

    # Generate plots
    plot_performance_comparison(benchmark_df, output_dir)
    plot_cpu_scaling(benchmark_df, output_dir)
    plot_throughput_comparison(benchmark_df, output_dir)
    plot_histogram_distribution(histogram_df, output_dir)

    print(f"Visualizations have been saved to: {output_dir}")

if __name__ == "__main__":
    main()