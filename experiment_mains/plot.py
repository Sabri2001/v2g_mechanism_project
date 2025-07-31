import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

# Load the reference image for dimension reference
ref_img_path = "screen_block.png"
ref_img = Image.open(ref_img_path)
ref_width, ref_height = ref_img.size

# Create horizontal SoC color bar (keeping 'SOC legend')
fig1, ax1 = plt.subplots(figsize=(6, 1))
cmap = plt.get_cmap('RdYlGn')
norm = Normalize(vmin=0, vmax=1)
cb1 = fig1.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax1, orientation='horizontal')
# cb1.ax.xaxis.set_ticks_position('top')
# cb1.ax.xaxis.set_label_position('top')
cb1.set_label('State of Charge (%)', fontsize=12)
cb1.set_ticks([0, 1])
cb1.set_ticklabels(['0 %', '100 %'])
fig1.tight_layout()
horizontal_cbar_path = "horizontal_soc_colorbar.png"
fig1.savefig(horizontal_cbar_path, dpi=300, bbox_inches='tight')
plt.close(fig1)


# Create a SoC block from x to y% in similar dimensions to the uploaded image
fig2, ax2 = plt.subplots(figsize=(ref_width / 100, ref_height / 100))
soc_data = np.linspace(0, 0.5, 100).reshape(1, -1)
ax2.imshow(soc_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
ax2.axis('off')
soc_block_path = "soc_block_0_to_50.png"
fig2.savefig(soc_block_path, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig2)
