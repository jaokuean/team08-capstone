# Exploring Portfolio Decarbonisation using AI

This project aims to utilise a machine learning approach involving the use of text mining techniques together with image processing techniques to extract information related to decarbonisation from companies’ reports. We will be extracting 3 main types of information: (1) text; (2) chart images; (3) tables. Our end product is a dashboard that will dislpay all extracted information. Furthermore, this dashboard is able to accept new PDF urls and run them through our entire pipeline for extraction. 


Our project pipeline is as follows: 

![project-pipeline.png](https://i.ibb.co/r4S0Bzv/Screenshot-2021-11-11-at-12-26-24-AM.png)



## Getting Started
This project uses Python 3.7.6, Jupyter Notebooks, Python Scripts and other open source packages that have to be installed for the code to run. The operating system will be a MacOS machine with intel processor chips. You can either clone this repository into your local machine or download all our codes and data from [our drive](https://drive.google.com/drive/folders/1ce9L5dHZXrWLzpRNf3iq6cdK5KU_QF0a?usp=sharing) into a single zip folder. Upon doing so, run the following commands within the root directory. 

```bash
conda create -n nus08_env python=3.7.6
conda activate nus08_env
while read requirement; do conda install -n nus08_env --y -q -c conda-forge -c pytorch -c anaconda -c ralexx $requirement || pip install $requirement; done < requirements.txt
```
This will create a conda environment named **nus08_env** and install all the relevant packages required for this project. This step will take some time as a hybrid requirements file incorporating conda and pip installs had to be generated for our unique dependency requirements. 

As there are additional files that are too big to upload to github, but are necessary to run the pipeline, you will also need to do the following steps:
1. Place the **data** and **assets** folder in the root folder
2. Place "model_final.pth" into the **table_extraction** folder
3. unzip the bert model "bert_model.zip" and place it inside the root folder
4. Edit the build.py file in the detectron2 package by substiuting "cfg.MODEL.DEVICE" to "cpu" in the codeline that starts with "model.to()". Skip this step if you are able to access GPU on your machine. The file is located at a path similar to this ~./opt/anaconda3/envs/env_name/lib/python3.7/site-packages/detectron2/modelling/meta_arch/build.py

Note : If you did not clone this repository but used the folder from our drive to obtain our codes and data, you can skip steps 1,2 and 3.

When done, activate the nus08_env and run this in the new terminal window to start the API connection with BERT-as-service. 
```bash 
conda activate nus08_env
bash shell_scripts/start_bert.sh
```

To run our dashboard which will access our entire information extraction pipeline, open another terminal, activate the same environment as above and run the following in the root directory:
```bash
python app.py
````

Afterwards, simply copy and paste the link that appears in your terminal into your browser. Click enter and you should be able to begin viewing all the processed decarbonisation related information. 

![DashboardLink.png](https://i.ibb.co/3BD9Mjc/tg-image-2356045203.jpg)


## Files
The following table contains a brief description of the files and folders in this repository.
| Folder / File | Description |
| - | - |
| **main.py** | Main file for running the entire project pipeline for a new report URL. |
| **app.py** | Main file for running dashboard which also runs the main.py if a new report URL is uploaded. |
| **requirements.txt** | This contains all the package and version requirements for our project. It has been modified to support both conda install and pip install as some of our packages cannot be found on conda channels |
| **shell_scripts** | This folder is used to contain shell scripts. start_bert.sh is used to initialise the connection with the BERT-as-service API |
| **chart_extraction** | Folder containing jupyter notebook that is used for internal analysis and code development of chart pipeline and python script that will run the chart extraction pipeline |
| **combining_data** | Folder containing jupyter notebook that will combine all information extracted to create final database |
| **relevance_prediction** | Folder containing jupyter notebook that is used for internal analysis and code development of machine learning models for relevance prediction and python script that will run the the relevance prediction portion of the text extraction pipeline  | 
| **sentiment_analysis** | Folder containing jupyter notebook that is used for internal analysis and code development of VADER to conduct sentiment analysis and python script that will run the the sentiment analysis portion of the text extraction pipeline  |
| **table_extraction** | Folder containing jupyter notebook that is used for internal analysis and code development of table pipeline, python script that will run the table extraction pipeline, yaml files and our modified Multi_Type_TD_TSR package that is required for the table extraction pipeline code to run |
| **text_classification** | Folder containing jupyter notebook that is used for internal analysis and code development of machine learning models for text classification and python script that will run the the text classification portion of the text extraction pipeline  | 
| **text_filtering** | Folder containing jupyter notebook that is used for internal analysis and code development of data collection, page and sentence filtering for subsequent tasks and python script that will run the data collection, page and sentence filtering pipeline  | 
| **word_cloud** | Folder containing jupyter notebook that is used for internal analysis and code development of word clouds and python script that will run the word cloud generation portion of the text extraction pipeline |
## Data
The following table contains a brief description of the files and folders our the data folders.
| Folder | Description |
| - | - |
| **assets** | Main data folder for data used in the dashboard. It contains the stylesheet, images and database used for the dashboard |
| **data** | Main data folder for data used during internal analysis and development. Main folders required for final pipeline to run includes **saved_models** and **new_report** folders. **saved_models** folder contains trained models required for the text extraction pipeline. **new_report** folder contains 3 empty folders namely: "ChartExtraction_Output", "wordcloud_images" and "table_images". This folder is used as an intermediate folder when a new report URL is uploaded. The information extracted from the new report after running through our whole pipeline will then be moved from the new_report folder to the main database in the assets/dashboard_data folder. |


## Application Demo
On our dashboard, users can search for a company name to obtain the relevant information extracted from a specific report, displayed on the dashboard. However if the desired report does not exist in our current database, users can upload a URL to the PDF or upload a PDF from their local directory. This new report will run through our pipeline, relevant information will extracted and displayed on the dashbaord, it will also be appended to our database for future use.
You can access a video of our dashboard demo here : https://youtu.be/VApBSNr_FFg

You can access a user guide for our dashboard here : https://docs.google.com/document/d/1Of3vbcNZCA1xCuofX5ji5_UvIn1hQWisSlWee9ZLsRU/edit?usp=sharing 

## Built With
- [plotly | dash](https://dash.plotly.com/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Natural Language Toolkit](https://www.nltk.org/)
- [spaCy](https://spacy.io/)
- [BERT-as-Service](https://github.com/hanxiao/bert-as-service)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/4.1.2/)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Camelot-Py](https://camelot-py.readthedocs.io/en/master/)
- [PyTesseract](https://github.com/tesseract-ocr/tesseract)
- [PyTorch](https://pytorch.org/)
- [pdfminer3](https://github.com/gwk/pdfminer3)
- [pdf2image](https://github.com/Belval/pdf2image)

## Authors
- Aw Xin Min - [Github](https://github.com/awxinmin)
- Chia Ai Fen - [Github](https://github.com/chiaaifen)
- Lim Jermaine - [Github](https://github.com/limjermaine88)
- Wong Jao Kuean - [Github](https://github.com/jaokuean)
- Lee Jun Hui Sean - [Github](https://github.com/seansljh)
