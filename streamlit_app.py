'''

This webui is designed to showcase the results in a user-friendly way. However, it uses CPU
as streamlit servers do not provide GPUs in its free service. Therefore, the inference here
is way slower. For GPU inference please clone the repo, install the packages ( you can find
it in requirements.txt) and 'pip install streamlit'. You can then run the streamlit webui on
your machine by 'streamlit run streamlit_app_gpu.py'. My machine's nividia computability ra-
ting is 6.1 and it runs lightning fast.

'''
import torch
import os
import gdown
import torch.nn.functional as F
import streamlit as st
from runpy import run_path
from PIL import Image, ImageFilter
from skimage.util import img_as_ubyte
from io import BytesIO
import numpy as np


device = 'cpu'
import torch
import os
import torch.nn.functional as F
import streamlit as st
from runpy import run_path
from PIL import Image, ImageFilter
from skimage.util import img_as_ubyte
from io import BytesIO
import numpy as np


def decrease_resolution(image, max_width, max_height):
    # Get the original width and height of the image
    width, height = image.size

    # Calculate the aspect ratio of the image
    aspect_ratio = width / height
    new_width = width
    new_height = height

    # Calculate the new width and height while maintaining the aspect ratio
    if width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    if height > max_height:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    return resized_image


def increase_resolution(image, target_width, target_height):
    # Get the original width and height of the image
    width, height = image.size

    # Calculate the aspect ratio of the image
    aspect_ratio = width / height

    # Calculate the new width and height while maintaining the aspect ratio
    if width < target_width:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    elif height < target_height:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        # No need to resize if the image is already larger than the specified dimensions
        return image

    # Resize the image while maintaining the aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    return resized_image

    # Create a new image with the target dimensions and paste the resized image onto it
    # final_image = Image.new("RGB", (target_width, target_height))
    # final_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    #
    # return final_image

#
# def rotation(image, width, height, prior=True):
#     if prior:
#         rotationbool = False
#         if height > width:
#             rotationbool = True
#             angle = 90
#             image = image.rotate(angle, expand=True)
#         return image, rotationbool
#     else:
#         angle = -90
#         image = image.rotate(angle, expand=True)
#         return image, True
#

# Loading the model Once for Optimization
@st.cache_resource
def loadModel():
    # Get model weights and parameters
    parameters = {'inp_channels': 3, 'out_channels': 3, 'dim': 48, 'num_blocks': [4, 6, 6, 8],
                  'num_refinement_blocks': 4,
                  'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66, 'bias': False, 'LayerNorm_type': 'WithBias',
                  'dual_pixel_task': False}
    weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    # Load Model Arch and convert to CPU
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
    model.cpu()
    # Load weights&Params to Model
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    print("************** Model Loaded **************")
    return model, weights


# Gets the download Link
def downloadImgBtn(img):
    # Original image came from cv2 format, fromarray convert into PIL format
    # Convert to Bytes
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return st.download_button(
        label="Download Restored Image",
        data=byte_im,
        file_name="restored.jpeg",
        mime="image/jpeg",
    )


if __name__ == '__main__':
    # Task
    task = 'Motion_Deblurring'

    # Model
    model, weights = loadModel()

    st.title("Text Image Enhancement")
    st.caption("An Assessment Task for Blnk")

    # Image Uploader
    st.subheader("Image Uploader:")
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    uploaded_file = st.file_uploader("Choose an image...", type=extensions)

    if uploaded_file is not None:
        # Displaying the image
        st.subheader("Uploaded Image")
        pilImg = Image.open(uploaded_file).convert('RGB')

        # Pre Processing
        target_width, target_height = pilImg.size
        max_width = 300
        max_height = 300
        pilImg = decrease_resolution(pilImg, max_width, max_height)

        # Displaying Image
        st.image(pilImg, caption='Uploaded Image', use_column_width=True)

        # # Rotation
        # pilImg, rot = rotation(pilImg, target_width, target_height, prior=True)

        # Give the user an on-screen indication that we are working
        onscreen = st.empty()
        onscreen.text('Restoring...')

        # Inference
        img_multiple_of = 8
        print(f"\n ==> Running {task} with weights {weights}\n ")
        with torch.no_grad():
            # Empty cuda cache
            # torch.cuda.ipc_collect()
            # torch.cuda.empty_cache()

            # load into cuda
            img = np.array(pilImg)
            input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cpu()

            # Pre inference processing Pad the input if not_multiple_of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (w + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - h if h % img_multiple_of != 0 else 0
            padw = W - w if w % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            # Inference
            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :h, :w]
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])

            # # Image Sharpening
            sharpen_filter = [
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            ]
            # Original Resolution
            im = Image.fromarray(restored.astype('uint8'), 'RGB')
            im = increase_resolution(im, target_width, target_height)
            # if rot:
            #     im = rotation(im, target_width, target_width, False)
            # Create a kernel filter using the defined matrix
            kernel_filter = ImageFilter.Kernel((3, 3), sharpen_filter)
            # Apply the kernel filter to the image
            restored = im.filter(kernel_filter)

            # Show the user that we have finished
            onscreen.empty()
            st.subheader("Restored Image")
            st.image(restored, caption='Restored Image', use_column_width=True)

            # Downloading Image Button
            btn = downloadImgBtn(restored)
