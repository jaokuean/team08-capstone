# Exploring Portfolio Decarbonisation using AI

This project aims to utilise a hybrid machine learning approach involving the use of text mining techniques together with image processing techniques to extract information related to decarbonisation from companies’ reports. We will be extracting 3 main types of information: (1) text; (2) chart images; (3) tables. Our end product is a dashboard that will dislpay all extracted information.


Our project pipeline is as follows: 
[![project-pipeline.png]("https://i.ibb.co/r4S0Bzv/Screenshot-2021-11-11-at-12-26-24-AM.png")](https://i.ibb.co/r4S0Bzv/Screenshot-2021-11-11-at-12-26-24-AM.png)



## Getting Started
This project uses Python 3.7.6, Jupyter Notebooks, Python Scripts and other open source pacakges that have to installed for the code to run. Upon cloning this repository into your local machine, run the following command to install most of the relevant packages.
```bash
conda create -n newenv python=3.7.6
conda activate newenv
while read requirement; do conda install --yes -c conda-forge -c pytorch -c anaconda -c ralexx $requirement || pip install $requirement; done < requirements.txt
```

As there are additional files that are too big to upload to github, but are required to run the pipeline, you will also need to do the following steps:
1. Place "model_final.pth" into the **table_detection** folder
2. unzip the bert model "uncased_L-12_H-768_A-12.zip" and place it inside the root folder
3. Change a file in the detectron2 package by commenting out the second code line that starts with "model.to()". Skip this step if you have GPU on your machine. The file is located at a path similar to this /Users/xinminaw/opt/anaconda3/envs/env_name/lib/python3.7/site-packages/detectron2/modelling/meta_arch/build.py

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
| **chart_extraction** | Folder containing jupyter notebook that is used for internal analysis and code development of chart pipeline and python script that will run the chart extraction pipeline |
| **combining_data** | Folder containing jupyter notebook that will combine all information extracted to create final database |
| **relevance_prediction** | Folder containing jupyter notebook that is used for internal analysis and code development of machine learning models for relevance prediction and python script that will run the the relevance prediction portion of the text extraction pipeline  | 
| **sentiment_analysis** | Folder containing jupyter notebook that is used for internal analysis and code development of VADER to conduct sentiment analysis and python script that will run the the sentiment analysis portion of the text extraction pipeline  |
| **table_extraction** | Folder containing jupyter notebook that is used for internal analysis and code development of table pipeline, python script that will run the table extraction pipeline, yaml files and our modified Multi_Type_TD_TSR pacakge that is required for the pipeline code to run. |
| **text_classification** | Folder containing jupyter notebook that is used for internal analysis and code development of machine learning models for text classification and python script that will run the the text classification portion of the text extraction pipeline  | 
| **text_filtering** | Folder containing jupyter notebook that is used for internal analysis and code development of data collection, page and sentence filtering for subsequent tasks and python script that will run the data collection, page and sentence filtering pipeline  | 
| **word_cloud** | Python script that will run the word cloud generation portion of the text extraction pipeline |


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
