import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

st.set_page_config(page_title='Greetings from Reddit')
st.title('Greetings from Reddit')

st.caption('_"\nIf you wanted to start an adult social network in 2022, you’d need to be web-only on iOS and side load on Android, take payment in crypto, have a way to convert crypto to fiat for business operations without being blocked, do a ton of work in age and identity verification and compliance so you don’t go to jail, protect all of that identity information so you don’t dox your users, and make a ton of money._"\n -Matt Mullenweg, CEO of Tumblr (2022)')

st.markdown("Greetings from Reddit is a set of LoRA weights trained on men uploading nudes onto Reddit with their own captions.")
st.markdown("It is a response to the increased santization of the Internet where adult spaces and content, sexually explicit or not, becomes increasingly hard to survive often due to monetization incentives.")
st.markdown("These captions are often suprisingly casual, even innocuous in nature.")
    
def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
    

@st.cache_resource
def load_model():
    model_base = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.unet.load_attn_procs("pytorch_lora_weights.bin")
    pipe.to("cuda")
    return pipe

model = load_model()




prompt_input = st.text_input("How would you greet the Internet?")

if(st.button('Submit')):
    prompt = prompt_input.title()
    #image = model(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    st.markdown("What Reddit is thinking:")
    col1,  col2, col3 = st.columns(3)
    
    with col1:
        image = model(prompt, num_inference_steps=30, guidance_scale=10,).images[0]
        st.image(image)
    
    with col2:
        image = model(prompt, num_inference_steps=30, guidance_scale=10,).images[0]
        st.image(image)
    
    with col3:
        image = model(prompt, num_inference_steps=30, guidance_scale=10,).images[0]
        st.image(image)
    