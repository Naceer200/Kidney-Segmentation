import os, glob
import numpy as np
from scipy import ndimage as ndi
import imageio
from scipy.ndimage import center_of_mass, zoom, measurements, binary_dilation
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage import measure
from skimage.measure import label, regionprops
import plotly.graph_objects as go
import cv2
import glob

datasets=imageio.volread("C:/Users/HI/PycharmProjects/pythonProject2/New data/data",format='dcm')
'''
def remove_trachea(slc, c=0.0069):
    new_slc = slc.copy()
    labels = label(slc,connectivity=1,background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas/512**2 > c)[0]
    for i in idxs:
        new_slc[tuple(rps[i].coords.T)] = 0
    return new_slc

def delete(masked):
    new_mask = masked.copy()
    labels = label(masked, background=0)
    idxs = np.unique(labels)[1:]
    COM_xs = np.array([center_of_mass(labels==i)[1] for i in idxs])
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y, COM_x in zip(idxs, COM_ys, COM_xs):
        if (COM_y < 0.50*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_y > 0.65*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x > 0.5*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x < 0.5*masked.shape[0]):
            new_mask[labels==idx] = 0
        else:
            new_mask[labels==idx] = 0
    return new_mask


def process_images(folder_path):
    image = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            original_image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            filt_m=ndi.median_filter(thresholded_image,size=10)
            filt_g=ndi.gaussian_filter(filt_m,sigma=2)
            threshold = 200    #610
            mask = filt_g > threshold
            masks = label(mask)
            maskss = remove_trachea(masks)
            #masks = clear_border(masks)
            image.append(maskss)
    return image
            # Display the results (optional)


# Specify the path to the folder containing JPG images
image_folder = "C:/Users/HI/PycharmProjects/pythonProject2/meningioma"

# Process the images in the specified folder
masked = process_images(image_folder)
#plt.pcolormesh(masked[15])
#plt.colorbar()
plt.imshow(masked[25], cmap="gray")
plt.show()
'''

#datasets = datasets.pixel_array
filt_m=ndi.median_filter(datasets,size=10)
filt_g=ndi.gaussian_filter(filt_m,sigma=2)
#hist=ndi.histogram(filt_g,min=0,max=65535,bins=65536)
#print(hist.shape)
#plt.plot(hist)
#plt.show()
#real = 60
threshold = 610     #610
# Apply thresholding to create a binary mask
mask = filt_g > threshold

def delete(masked):
    new_mask = masked.copy()
    labels = label(masked, background=0)
    idxs = np.unique(labels)[1:]
    COM_xs = np.array([center_of_mass(labels==i)[1] for i in idxs])
    COM_ys = np.array([center_of_mass(labels==i)[0] for i in idxs])
    for idx, COM_y, COM_x in zip(idxs, COM_ys, COM_xs):
        if (COM_y < 0.40*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_y > 0.80*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x > 0.5*masked.shape[0]):
            new_mask[labels==idx] = 0
        elif (COM_x < 0.40*masked.shape[0]):
            new_mask[labels==idx] = 0
        else:
            new_mask[labels==idx] = 1
    return new_mask


masks = np.vectorize(clear_border, signature='(n,m)->(n,m)')(mask)
maskss = np.vectorize(ndi.binary_opening, signature='(n,m)->(n,m)')(masks)
masksss = np.vectorize(ndi.binary_closing, signature='(n,m)->(n,m)')(maskss)

new_mask = np.vectorize(delete, signature='(n,m)->(n,m)')(masksss)
nmask = np.vectorize(ndi.binary_fill_holes, signature='(n,m)->(n,m)')(new_mask)
masked = binary_dilation(nmask, iterations=5)
#masked = np.stack(maskeds, axis=0)
'''
output_folder = "C:/Users/HI/PycharmProjects/pythonProject2/New data/brainM"

def save_segmented_slices(images, output_folder= output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, image in enumerate(images):
        filename = os.path.join(output_folder, f"1-{i+1}.dcm")
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    #output = output_folder
    save_segmented_slices(masked)
'''
im = zoom(1*(masked), (0.3,0.3,0.3))
z, y, x = [np.arange(i) for i in im.shape]
z*=4
X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=np.transpose(im,(1,2,0)).flatten(),
    isomin=0.1,
    opacity=0.1,
    surface_count=17,
    ))
fig.write_html("testss.html")

fig,axes=plt.subplots(nrows=2,ncols=3)
# Display the original image
axes[0,0].imshow(datasets[118], cmap="gray")
axes[0,0].set_title('slice 118')
axes[0,0].axis('off')
axes[0,1].imshow(datasets[135],cmap='gray')
axes[0,1].set_title('slice 135')
axes[0,1].axis('off')
axes[0,2].imshow(datasets[145],cmap='gray')
axes[0,2].set_title('slice 145')
axes[0,2].axis('off')
axes[1,0].imshow(masked[118],cmap='gray')
axes[1,0].set_title('mask 118')
axes[1,0].axis('off')
axes[1,1].imshow(masked[135],cmap='gray')
axes[1,1].set_title('mask 135')
axes[1,1].axis('off')
axes[1,2].imshow(masked[145],cmap='gray')
axes[1,2].set_title('mask 145')
axes[1,2].axis('off')
#plt.imshow(masked[61], cmap="gray")
#plt.colorbar()
plt.show()


