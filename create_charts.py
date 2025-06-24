import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df = pd.read_csv('results/results.csv')

# Clean up column names (remove spaces)
df.columns = df.columns.str.strip()

# Separate color and grayscale data
color_data = df[df['num_of_channels'] == 3].copy()
grayscale_data = df[df['num_of_channels'] == 1].copy()

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CUDA Box Filter Performance Analysis', fontsize=16, fontweight='bold')

# Helper function to format image names with dimensions
def format_image_name_with_size(name, dimensions):
    formatted_name = name.replace('.jpg', '').replace('_', ' ').title()
    return f'{formatted_name}\n({dimensions})'

def format_image_name(name):
    return name.replace('.jpg', '').replace('_', ' ').title()

# Chart 1: Color Images - CPU vs GPU Time
color_images = [format_image_name_with_size(img, dim) for img, dim in zip(color_data['image_name'], color_data['image_dimensions'])]
x_pos = np.arange(len(color_images))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, color_data['cpu_time(ms)'], width, 
                label='CPU', color='#ff6b6b', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, color_data['gpu_time(ms)'], width, 
                label='GPU (CUDA)', color='#4ecdc4', alpha=0.8)

ax1.set_title('Color Images (3 Channels) - Processing Time', fontweight='bold')
ax1.set_xlabel('Test Images')
ax1.set_ylabel('Processing Time (ms)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(color_images, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Chart 2: Grayscale Images - CPU vs GPU Time
grayscale_images = [format_image_name_with_size(img, dim) for img, dim in zip(grayscale_data['image_name'], grayscale_data['image_dimensions'])]
x_pos = np.arange(len(grayscale_images))

bars3 = ax2.bar(x_pos - width/2, grayscale_data['cpu_time(ms)'], width, 
                label='CPU', color='#ff6b6b', alpha=0.8)
bars4 = ax2.bar(x_pos + width/2, grayscale_data['gpu_time(ms)'], width, 
                label='GPU (CUDA)', color='#4ecdc4', alpha=0.8)

ax2.set_title('Grayscale Images (1 Channel) - Processing Time', fontweight='bold')
ax2.set_xlabel('Test Images')
ax2.set_ylabel('Processing Time (ms)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(grayscale_images, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Chart 3: Color Images - Speedup Factor
bars5 = ax3.bar(color_images, color_data['speedup'], 
                color='#45b7d1', alpha=0.8, edgecolor='darkblue')
ax3.set_title('Color Images - GPU Speedup Factor', fontweight='bold')
ax3.set_xlabel('Test Images')
ax3.set_ylabel('Speedup (×)')
ax3.set_xticklabels(color_images, rotation=45)
ax3.grid(True, alpha=0.3)

# Add speedup values on bars
for i, bar in enumerate(bars5):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Chart 4: Grayscale Images - Speedup Factor
bars6 = ax4.bar(grayscale_images, grayscale_data['speedup'], 
                color='#96ceb4', alpha=0.8, edgecolor='darkgreen')
ax4.set_title('Grayscale Images - GPU Speedup Factor', fontweight='bold')
ax4.set_xlabel('Test Images')
ax4.set_ylabel('Speedup (×)')
ax4.set_xticklabels(grayscale_images, rotation=45)
ax4.grid(True, alpha=0.3)

# Add speedup values on bars
for i, bar in enumerate(bars6):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{height:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('results/performance_analysis.pdf', bbox_inches='tight')

# Create a second figure focusing on the dramatic time differences
fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('CPU vs GPU Processing Time Comparison', fontsize=16, fontweight='bold')

# Combined comparison for color images (log scale for better visualization)
color_images_log = [format_image_name_with_size(img, dim) for img, dim in zip(color_data['image_name'], color_data['image_dimensions'])]
ax5.bar(x_pos - width/2, color_data['cpu_time(ms)'], width, 
        label='CPU', color='#ff6b6b', alpha=0.8)
ax5.bar(x_pos + width/2, color_data['gpu_time(ms)'], width, 
        label='GPU (CUDA)', color='#4ecdc4', alpha=0.8)
ax5.set_title('Color Images - Time Comparison (Log Scale)', fontweight='bold')
ax5.set_xlabel('Test Images')
ax5.set_ylabel('Processing Time (ms)')
ax5.set_yscale('log')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(color_images_log, rotation=45)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Combined comparison for grayscale images (log scale)
grayscale_images_log = [format_image_name_with_size(img, dim) for img, dim in zip(grayscale_data['image_name'], grayscale_data['image_dimensions'])]
x_pos_gray = np.arange(len(grayscale_images_log))
ax6.bar(x_pos_gray - width/2, grayscale_data['cpu_time(ms)'], width, 
        label='CPU', color='#ff6b6b', alpha=0.8)
ax6.bar(x_pos_gray + width/2, grayscale_data['gpu_time(ms)'], width, 
        label='GPU (CUDA)', color='#4ecdc4', alpha=0.8)
ax6.set_title('Grayscale Images - Time Comparison (Log Scale)', fontweight='bold')
ax6.set_xlabel('Test Images')
ax6.set_ylabel('Processing Time (ms)')
ax6.set_yscale('log')
ax6.set_xticks(x_pos_gray)
ax6.set_xticklabels(grayscale_images_log, rotation=45)
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/time_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('results/time_comparison.pdf', bbox_inches='tight')

print("Charts created successfully!")
print("Files generated:")
print("- results/performance_analysis.png (comprehensive analysis)")
print("- results/performance_analysis.pdf")
print("- results/time_comparison.png (log scale time comparison)")
print("- results/time_comparison.pdf")

# Display summary statistics
print(f"\nPerformance Summary:")
print(f"Color Images - Average Speedup: {color_data['speedup'].mean():.1f}×")
print(f"Grayscale Images - Average Speedup: {grayscale_data['speedup'].mean():.1f}×")
print(f"Overall Average Speedup: {df['speedup'].mean():.1f}×")
print(f"Best Speedup: {df['speedup'].max():.1f}× ({df.loc[df['speedup'].idxmax(), 'image_name']})") 