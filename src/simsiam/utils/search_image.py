import requests
from PIL import Image
from io import BytesIO
import os
import logging

def search_image(cfg, rd, imagePath, img_id):
    BASE_URI = 'https://api.bing.microsoft.com/v7.0/images/visualsearch'
    SUBSCRIPTION_KEY = "Your Own Key"
    HEADERS = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY}
    file = {'image' : ('myfile', open(imagePath, 'rb'))}

    logging.info(imagePath)
    # additional_path = cfg.dataset.name+'/additional/'+cfg.mlflow_runname+'/round'+str(rd)+'/sub/'
    additional_path = cfg.dataset.name+'/additional/'+cfg.mlflow_runname+'/'+str(cfg.train_parameters.seed)+'/round'+str(rd)+'/sub/'
    try:
        response = requests.post(BASE_URI, headers=HEADERS, files=file)
        response.raise_for_status()

        search_results = response.json()
        thumbnail_urls = [img["thumbnailUrl"] for img in search_results['tags'][0]['actions'][2]['data']['value'][:cfg.train_parameters.num_images]]
        
        os.makedirs(additional_path, exist_ok=True)
        for url in thumbnail_urls:
            img_id += 1
            image_data = requests.get(url)
            image_data.raise_for_status()
            image = Image.open(BytesIO(image_data.content)) 
            image.save(additional_path+str(img_id)+'.png')
        return img_id
    
    except:
        pass
    