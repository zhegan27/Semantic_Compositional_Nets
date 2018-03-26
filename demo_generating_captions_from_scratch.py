"""
Demo for Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zhe.gan@duke.edu, March, 23, 2018
"""

from os import system
import urllib.request
from image_captioning import caption_generation
from utilities import feature_loading
import argparse
import pickle
import datetime

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='Image Captioning ')
    parser.add_argument( "--image_url", help="the url of the input image" , default='http://www.trainyourpup.biz/cm/dpl/images/create/frisbee1.jpg')
    args = parser.parse_args()
    
    print('start image captioning @ '+str(datetime.datetime.now().time()))
    
    image_name = args.image_url.split('/')[-1]
    
    print("first, downloading the image ...")
    urllib.request.urlretrieve(args.image_url, "./Images/"+image_name)
    
    file = open("temp.lst","w")
    file.write("Images/"+image_name)
    file.close()
    
    x = pickle.load(open("./pretrained_model/tag_vocab.p","rb"))
    tag_wordtoix, tag_ixtoword = x[0],x[1]
    del x    

    # This step requirs windows system
    print("second, extract image features ...")
    system('.\\bin\Release\demo.exe /m model /i temp.lst') # feature extractor
    # img_feats and tag_feats are used for generating captions
    img_feats, tag_feats, tags = feature_loading(tag_wordtoix, tag_ixtoword) 
    
    print("Now, start image captioning ...")
    N = 6 # define how many ensembles to use
    predtext = caption_generation(N, img_feats, tag_feats)
    
    print("Detected tags: "+tags)
    print("Generated captions: "+predtext[0][0])
    
    print('end @ '+str(datetime.datetime.now().time()))

    
