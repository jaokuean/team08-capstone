# Exploring Portfolio Decarbonisation using AI

This project aims to utilise a hybrid machine learning approach involving the use of text mining techniques together with image processing techniques to extract information related to decarbonisation from companies’ reports. We will be extracting 3 main types of information: (1) text; (2) chart images; (3) tables. Our end product is a dashboard that will dislpay all extracted information.


Our project pipeline is as follows: 
[![project-pipeline.png]("https://i.ibb.co/r4S0Bzv/Screenshot-2021-11-11-at-12-26-24-AM.png")](https://i.ibb.co/r4S0Bzv/Screenshot-2021-11-11-at-12-26-24-AM.png)



## Getting Started
This project uses Python 3.7.6, Jupyter Notebooks, Python Scripts and other open source pacakges that have to installed for the code to run. Upon cloning this repository into your local machine, run the following command to create a conda environment and install most of the relevant packages.
```bash
conda create -n newenv python=3.7.6
conda activate newenv
while read requirement; do conda install --yes -c conda-forge -c pytorch -c anaconda -c ralexx $requirement || pip install $requirement; done < requirements.txt
```

As there are additional files that are too big to upload to github, but are necessary to run the pipeline, you will also need to do the following steps:
1. Place the **data** folder in the root folder
2. Place "model_final.pth" into the **table_extraction** folder
3. unzip the bert model "uncased_L-12_H-768_A-12.zip" and place it inside the root folder
4. Edit the build.py file in the detectron2 package by substiuting "cfg.MODEL.DEVICE" to "cpu" in the codeline that starts with "model.to()". Skip this step if you are able to access GPU on your machine. The file is located at a path similar to this /Users/xinminaw/opt/anaconda3/envs/env_name/lib/python3.7/site-packages/detectron2/modelling/meta_arch/build.py

Note : If you did not clone this repository but used the zip file specified in our report in obtain our codes and data, you can skip steps 1,2 and 3.

When done, run this in the environment to start the bert model.
```bash
bert-serving-start -model_dir /path to your bert model/ -num_worker=4
```

To run our dashboard which will access our project pipeline (from data collection to all information extraction), open another terminal and activate the same environment as above and run the following in the root directory:
```bash
python app.py
````


## Files
The following table contains a brief description of the files and folders in this repository.
| Folder / File | Description |
| - | - |
| **main.py** | Main file for running the entire project pipeline for a new report URL. |
| **app.py** | Main file for running dashboard which also runs the main.py if a new report URL is uploaded. |
| **requirements.sh** | KIV |
| **requirements.txt** | KIV |
| **chart_extraction** | Folder containing jupyter notebook that is used for internal analysis and code development of chart pipeline and python script that will run the chart extraction pipeline |
| **combining_data** | Folder containing jupyter notebook that will combine all information extracted to create final database |
| **relevance_prediction** | Folder containing jupyter notebook that is used for internal analysis and code development of machine learning models for relevance prediction and python script that will run the the relevance prediction portion of the text extraction pipeline  | 
| **sentiment_analysis** | Folder containing jupyter notebook that is used for internal analysis and code development of VADER to conduct sentiment analysis and python script that will run the the sentiment analysis portion of the text extraction pipeline  |
| **table_extraction** | Folder containing jupyter notebook that is used for internal analysis and code development of table pipeline, python script that will run the table extraction pipeline, yaml files and our modified Multi_Type_TD_TSR package that is required for the table extraction pipeline code to run. |
| **text_classification** | Folder containing jupyter notebook that is used for internal analysis and code development of machine learning models for text classification and python script that will run the the text classification portion of the text extraction pipeline  | 
| **text_filtering** | Folder containing jupyter notebook that is used for internal analysis and code development of data collection, page and sentence filtering for subsequent tasks and python script that will run the data collection, page and sentence filtering pipeline  | 
| **word_cloud** | Python script that will run the word cloud generation portion of the text extraction pipeline |
## Data
| **assets** | Main data folder for data used in the dashboard. It conatins the stylesheet, images and database used for the dashboard. |
| **data** | Main data folder for data used during internal analysis and development. Main folders required for final pipeline to run includes **saved_models** and **new_report** folders. **saved_models** folder contains trained models required for the text extraction pipeline. **new_report** folder contains 3 empty folders namely: "ChartExtraction_Output", "wordcloud_images" and "table_images". This folder is used as an intermediate folder when a new report URL is uploaded. The information extracted from the new report after running through our whole pipeline will then be added to the main database in the assets/dashboard_data folder. |



## Application Demo
On our dashboard, users can search for a company name to obtain the relevant information extracted from a specific report, displayed on the dashboard. However if the desired report does not exist in our current database, users can upload a URL to the PDF or upload a PDF from their local directory. This new report will run through our pipeline, relevant information will extracted and displayed on the dashbaord, it will also be appended to our database for future use.
![burpple_plus_demo.gif](assets/burpple_plus_demo.gif) CHANGE

## Built With (help with this)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Natural Language Toolkit](https://www.nltk.org/)
- [plotly | dash](https://dash.plotly.com/)
- [BERT-as-Service](https://github.com/hanxiao/bert-as-service)
- [spaCy](https://spacy.io/)

## Authors
- Aw Xin Min - [Github](https://github.com/awxinmin)
- Chia Ai Fen - [Github](https://github.com/chiaaifen)
- Lim Jermaine - [Github](https://github.com/limjermaine88)
- Wong Jao Kuean - [Github](https://github.com/jaokuean)
- Lee Jun Hui Sean - [Github](https://github.com/seansljh)
