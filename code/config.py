import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = os.path.join(os.path.dirname('/media/ailab/songyoungtak/vcr1/vcr1images/'))
VCR_ANNOTS_DIR = os.path.join(os.path.dirname('/media/ailab/songyoungtak/vcr_new/new/add_keyword/'))
#VCR_IMAGE_RESIZE = os.path.join(os.path.dirname('/media/ailab/songyoungtak/vcr1/vcr1images_resized/'))

if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")
