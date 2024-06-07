import cc3d
import numpy as np

labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = cc3d.connected_components(labels_in) # 26-connected

connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

# If you need a particular dtype you can specify np.uint16, np.uint32, or np.uint64
# You can go bigger, not smaller, than the default which is selected
# to be the smallest that can be safely used. This can save you the copy
# operation needed by labels_out.astype(...).
labels_out = cc3d.connected_components(labels_in, out_dtype=np.uint64)

# If you're working with continuously valued images like microscopy
# images you can use cc3d to perform a very rough segmentation. 
# If delta = 0, standard high speed processing. If delta > 0, then
# neighbor voxel values <= delta are considered the same component.
# The algorithm can be 2-10x slower though. Zero is considered
# background and will not join to any other voxel.
labels_out = cc3d.connected_components(labels_in, delta=10)

# You can extract the number of labels (which is also the maximum 
# label value) like so:
labels_out, N = cc3d.connected_components(labels_in, return_N=True) # free
# -- OR -- 
labels_out = cc3d.connected_components(labels_in) 
N = np.max(labels_out) # costs a full read

# You can extract individual components using numpy operators
# This approach is slow, but makes a mutable copy.
for segid in range(1, N+1):
  extracted_image = labels_out * (labels_out == segid)
  process(extracted_image) # stand in for whatever you'd like to do

# If a read-only image is ok, this approach is MUCH faster
# if the image has many contiguous regions. A random image 
# can be slower. binary=True yields binary images instead
# of numbered images.
for label, image in cc3d.each(labels_out, binary=False, in_place=True):
  process(image) # stand in for whatever you'd like to do

# Image statistics like voxel counts, bounding boxes, and centroids.
stats = cc3d.statistics(labels_out)

# Remove dust from the input image. Removes objects with
# fewer than `threshold` voxels.
labels_out = cc3d.dust(
  labels_in, threshold=100, 
  connectivity=26, in_place=False
)

# Get a labeling of the k largest objects in the image.
# The output will be relabeled from 1 to N.
labels_out, N = cc3d.largest_k(
  labels_in, k=10, 
  connectivity=26, delta=0,
  return_N=True,
)
labels_in *= (labels_out > 0) # to get original labels

# Compute the contact surface area between all labels.
# Only face contacts are counted as edges and corners
# have zero area. To get a simple count of all contacting
# voxels, set `surface_area=False`. 
# { (1,2): 16 } aka { (label_1, label_2): contact surface area }
surface_per_contact = cc3d.contacts(
  labels_out, connectivity=connectivity,
  surface_area=True, anisotropy=(4,4,40)
)
# same as set(surface_per_contact.keys())
edges = cc3d.region_graph(labels_out, connectivity=connectivity)

# You can also generate a voxel connectivty graph that encodes
# which directions are passable from a given voxel as a bitfield.
# This could also be seen as a method of eroding voxels fractionally
# based on their label adjacencies.
# See help(cc3d.voxel_connectivity_graph) for details.
graph = cc3d.voxel_connectivity_graph(labels, connectivity=connectivity)