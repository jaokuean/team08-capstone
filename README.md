# Exploring Portfolio Decarbonisation using AI

This project aims to utilise a hybrid machine learning approach involving the use of text mining techniques together with image processing techniques to extract information related to decarbonisation from companies’ reports. We will be extracting 3 main types of information: (1) text; (2) chart images; (3) tables. Our end product is a dashboard that will dislpay all extracted information.


Our project pipeline is as follows: UPLOAD PHOTO



## Getting Started
This project uses Python 3.7.6, Jupyter Notebooks and Python Scripts. Upon cloning this repository into your local machine, run the following command to install all relevant packages.
```bash
pip install -r requirements.txt
```
To run our dashboard which will access our project pipeline (from data collection to all information extraction), run the following in the root directory:
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
- [Facebook fastText](https://fasttext.cc/)
- [Google BERT](https://arxiv.org/abs/1810.04805)
- [LIME](https://lime-ml.readthedocs.io/en/latest/)
- [plotly | dash](https://dash.plotly.com/)

## Authors
- Aw Xin Min - [Github](https://github.com/awxinmin)
- Chia Ai Fen - [Github](https://github.com/chiaaifen)
- Lim Jermaine - [Github](https://github.com/limjermaine88)
- Wong Jao Kuean - [Github] (https://github.com/jaokuean)
- Lee Jun Hui Sean - [Github](https://github.com/seansljh)
