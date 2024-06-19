import os  # Imports the os module to interact with the operating system.
import numpy as np  # Imports the numpy module for efficient numerical operations.
import pandas as pd  # Imports the pandas module for data manipulation in DataFrame format.
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # Imports TF-IDF and CountVectorizer text vectorization tools from scikit-learn.
from sklearn.model_selection import train_test_split  # Imports the train_test_split function from scikit-learn to split data into training and testing sets.
from sklearn.naive_bayes import MultinomialNB  # Imports the Multinomial Naive Bayes classifier from scikit-learn.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Imports metrics to evaluate model performance.
from sklearn.pipeline import Pipeline  # Imports the Pipeline class from scikit-learn to chain multiple data processing steps.
from nltk.corpus import stopwords  # Imports the NLTK stopwords list for text preprocessing.
from nltk.stem import RSLPStemmer  # Imports the RSLP stemmer from NLTK for reducing words to their root form.
from nltk.tokenize import word_tokenize  # Imports the NLTK word tokenizer to split text into tokens.
#from PyPDF2 import PdfReader  # Imports the PdfReader class from PyPDF2 to extract text from PDF files.
from tqdm import tqdm  # Imports the tqdm function to display progress bars during loops.
import pickle
import fitz  # PyMuPDF
from flask import Flask, Response, abort, jsonify, request
import nltk
import shutil

#curl -X POST http://SEU IP:5000/treino
#curl -X POST http://SEU IP:5000/classificar
#curl -X POST http://SEU IP:5000/ajustar

#ENDPOINTS treino, classificar, ajustar

app = Flask(__name__)

#para fazer o download das stopwords, apos o download pode ser comentado ########################################################################################
# nltk.download() 

stopwords = stopwords.words('portuguese')  # Gets the list of stopwords in Portuguese.
stemmer = RSLPStemmer()  # Initializes the RSLP stemmer for reducing words to their root form.



def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyMuPDF.

    Parameters:
    file_path (str): The path of the PDF file.

    Returns:
    str: The extracted text from the PDF file.
    """
    text = ""  # Inicializa uma string vazia para armazenar o texto extraído.
    try:
        with open(file_path, 'rb') as file:  # Abre o arquivo PDF no modo de leitura binária.
            pdf_document = fitz.open(file)  # Abre o documento PDF com PyMuPDF.
            for page_num in range(pdf_document.page_count):  # Itera sobre todas as páginas do PDF.
                page = pdf_document.load_page(page_num)  # Carrega a página atual do PDF.
                text += page.get_text()  # Extrai o texto da página atual e concatena-o à string de texto.
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    return text  # Retorna o texto extraído do PDF.


def preprocess_text(text):  # Defines a function for text preprocessing.
    """
    Performs text preprocessing.

    Parameters:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    words = word_tokenize(text, language='portuguese')  # Tokenizes the text into words.
    words = [stemmer.stem(word) for word in words if word not in stopwords]  # Applies stemming and removes stopwords.
    return ' '.join(words)  # Returns the preprocessed text as a single string.


def save_data(X_train, X_test, y_train, y_test, file_path):
    """
    Saves the model training and testing data to a file.

    Parameters:
    X_train (DataFrame): The training documents.
    X_test (DataFrame): The testing documents.
    y_train (DataFrame): The training labels.
    y_test (DataFrame): The testing labels.
    file_path (str): The file path to save the data.
    """
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    # Creates a dictionary containing the training and testing data
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        # Saves the data to the specified file using pickle
    print(f'Data saved to {file_path}')
    # Displays a message indicating that the data has been saved successfully





@app.route('/treino', methods=['POST'])
def resposta():
    print('HELLO WORLD')
    sector_mapping = {  # Defines a dictionary to map alphabetical sector codes to numbers.
    'PAS' :  0,
    'PDA' :  1,
    'PPE' :  2,
    'PSE' :  3,
    'PTR' :  4,
    'PUMA':  5,
    'PTA' :  6
    }

    # Directory where the data (PDFs) are stored
    data_dir = r'C:CAMINHO PARA O DIRETORIO DIRTREIN'  # Defines the directory where the PDF files are stored.
    pdf_files = os.listdir(data_dir)  # Lists the files in the specified directory.
    documents, labels = [], []
    # documents = []  # Initializes a list to store the documents (text extracted from PDFs).
    # labels = []  # Initializes a list to store the document labels (mapped sector codes).
    
   

    for file in tqdm(pdf_files, desc='Processing PDFs'):  # Iterates over the PDF files in the directory.
        if file.endswith('.pdf'):  # Checks if the file is a PDF.
            pdf_path = os.path.join(data_dir, file)  # Gets the full path of the PDF file.
            text = extract_text_from_pdf(pdf_path)  # Extracts text from the PDF.
            documents.append(text)  # Adds the extracted text to the list of documents.
            sector_code = file.split('_')[0]  # Extracts the sector code from the file name.
            sector_label = sector_mapping.get(sector_code)  # Gets the sector number mapped to the code.
            if sector_label is not None:  # Checks if the sector code is valid.
                labels.append(sector_label)  # Adds the label to the list of labels.
            else:
                print(f'Warning: Invalid sector code found in file {file}')  # Warning if the sector code is invalid.

    # Creating a DataFrame with the two training variables
    df = pd.DataFrame({'documents': documents,  # Creates a DataFrame with the documents.
                    'labels': labels  # Adds the labels to the DataFrame.
                    })

    X = df['documents']  # Extracts the documents as independent variables.
    y = df['labels']  # Extracts the labels as dependent variables.

    # Pipeline for vectorization/training model definition
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=10000)),  # Vectorization step using CountVectorizer.
        ('clf', MultinomialNB())  # Classification step using Multinomial Naive Bayes.
    ])

    

    print('Preprocessing training data...')  # Displays a message indicating the start of preprocessing.
    X_prep = X.apply(preprocess_text)  # Applies preprocessing to the documents.
    print('Preprocessing completed!')  # Displays a message indicating the completion of preprocessing.

    # Training the Naive Bayes model
    pipeline.fit(X_prep, y)  # Trains the model with the preprocessed documents and corresponding labels.

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_prep, labels, test_size=0.2, random_state=42)
    
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    file_path = 'trainingDataNaiveBayes.pkl'
    save_data(X_train, X_test, y_train, y_test, file_path)
    # Saves the model training and testing data to a file
    print('\n')
    print('Model training data saved')


    return jsonify({'message': 'Model trained successfully!', 'classification_report': report})



##########################################################################################################################################
#CLASSIFICACAO  #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO 
#CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO
#CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO 
#CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO 
#CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO #CLASSIFICACAO
##########################################################################################################################################


# Function to extract text from a PDF
def extrair_texto_classificacao(file_path_class):
    """
    Extracts text from a PDF file using PyMuPDF.

    Parameters:
    file_path (str): The path of the PDF file.

    Returns:
    str: The extracted text from the PDF file.
    """
    text = ""  # Initializing an empty string to store the extracted text.
    try:
        with fitz.open(file_path_class) as pdf_file:  # Open the PDF file.
            for page in pdf_file:  # Iterate over all pages of the PDF.
                text += page.get_text()  # Extract text from each page and concatenate it to the text string.
    except Exception as e:
        print(f"An error occurred: {e}")
    return text  # Returning the extracted text from the PDF.



def preprocess_classificacao(text_class):
    """
    Performs text preprocessing.

    Parameters:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    words = word_tokenize(text_class, language='portuguese')  # Tokenizing the text into words.
    words = [stemmer.stem(word) for word in words if word not in stopwords]  # Applying stemming and removing stopwords.
    return ' '.join(words)  # Returning the preprocessed text as a single string.

# Retrieving model training data
def load_data_classificacao(file_path_class):
    """
    Loads model training data from a file.

    Parameters:
    file_path (str): The path of the file containing training data.

    Returns:
    dict: A dictionary containing the training data.
    """
    with open(file_path_class, 'rb') as file:  # Opening the file in binary read mode.
        data = pickle.load(file)  # Loading data from the file using pickle.
    return data  # Returning the loaded data.

##########################################################################################################################################
#  Function predict  the  especialized from a PDF file
def predict_classificacao(file_path_class,switch_case_class):
    initialPetition = extrair_texto_classificacao(file_path_class)  # Extracting text from the PDF file.

    print('\n')

#     if initialPetition is not None:  # Checking if the text was successfully extracted.
    if not (initialPetition == ''): # Checking if the text was successfully extracted. (Changed at 22/05/2024) 
#         print('Preprocessing information for prediction...')  # Displaying a message indicating the start of preprocessing.
        X_prediction = [initialPetition]  # Putting the petition text into a list.
        prediction_preprocessing = preprocess_classificacao(X_prediction[0])  # Preprocessing the petition text.
#         print('Preprocessing of information for prediction completed!')  # Displaying a message indicating the completion of preprocessing.
        prediction = pipeline_class.predict([prediction_preprocessing])  # Making a prediction of the case outcome with the trained model.    
        
        # # Getting the specialization based on the value of prediction[0]
        specialized = switch_case_class.get(prediction[0], 'NOT FOUND')
# 
#         print("Directed towards the specialized :", specialized)
        return(specialized)
     
    else:
#         print("Failed to convert PDF to text.")  # Displaying a failure message if the PDF to text conversion fails.
        return("Failed to convert PDF to text.")



#para fazer o download das stopwords, apos o download pode ser comentado ########################################################################################
# nltk.download() 
# stopwords = stopwords.words('portuguese')  # Gets the list of stopwords in Portuguese.
# stemmer = RSLPStemmer()  # Initializes the RSLP stemmer for reducing words to their root form.

# Pipeline for vectorization/training model definition
pipeline_class = Pipeline([
        ('vect', CountVectorizer(max_features=10000)),  # Vectorization step using CountVectorizer.
        ('clf', MultinomialNB())  # Classification step using Multinomial Naive Bayes.
    ])



pdf_file_path_class = r"C:CAMINHO PARA O PDF"


@app.route('/classificar', methods=['POST'])
def resposta2():
    # data = request.get_json()
    # pdf_file_path = data['caminho']
    # Example usage:
    loaded_data = load_data_classificacao('trainingDataNaiveBayes.pkl')  # Loading training data from the file.
    X_train, X_test, y_train, y_test = loaded_data['X_train'], loaded_data['X_test'], loaded_data['y_train'], loaded_data['y_test']
    # Extracting training and testing sets from the loaded data.

    # print('\n')
    # print('Retrieved training data')  # Displaying a message indicating that the data has been successfully retrieved.

    # Fitting the pipeline to the training data
    pipeline_class.fit(X_train, y_train)  # Fitting the pipeline to the training data.

    # Initializing the dictionary with default values
    switch_case = {
        0: 'PAS',
        1: 'PDA',
        2: 'PPE',
        3: 'PSE',
        4: 'PTR',
        5: 'PUMA',
        6: 'PTA',
    }

    # while True:
    # # Predicting a new initial petition
    print("\nReviewing a new initial petition \n")
    
        # Asking the user for the name of the PDF file containing the new initial petition.
        
    print(predict_classificacao(pdf_file_path_class,switch_case))
    
    return jsonify({'message': 'Model trained successfully!', 'classification_report': predict_classificacao(pdf_file_path_class,switch_case)})






###################################################################################################################################################################
#COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO 
#COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO 
#COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO 
#COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO 
#COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO 
#COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO #COLOCAR DOCUMENTO NO LUGAR CERTO 
###################################################################################################################################################################


def copiar_pdf_para_diretorio(origem_pdf_erro, destino_diretorio_erro, especializada_erro):
    """
    Copia um arquivo PDF para um diretório específico, acrescentando um tipo ao nome do arquivo.

    Argumentos:
    origem_pdf (str): Caminho do arquivo PDF de origem.
    destino_diretorio (str): Caminho do diretório de destino.
    tipo (str): Tipo a ser acrescentado ao nome do arquivo.
    """
    # Verifica se o arquivo PDF de origem existe
    if not os.path.isfile(origem_pdf_erro):
        print(f"O arquivo {origem_pdf_erro} não existe.")
        return
    
    # Verifica se o destino é um diretório válido
    if not os.path.isdir(destino_diretorio_erro):
        print(f"O diretório {destino_diretorio_erro} não existe.")
        return

    try:
        # Extrai o nome do arquivo PDF
        nome_arquivo, extensao = os.path.splitext(os.path.basename(origem_pdf_erro))
        
        # Novo nome do arquivo com o tipo acrescentado
        novo_nome_arquivo = f"{especializada_erro}_{nome_arquivo}{extensao}"
        
        # Copia o arquivo PDF para o diretório de destino com o novo nome
        shutil.copy(origem_pdf_erro, os.path.join(destino_diretorio_erro, novo_nome_arquivo))
        print(f"Arquivo PDF copiado com sucesso para {os.path.join(destino_diretorio_erro, novo_nome_arquivo)}")
    except Exception as e:
        print(f"Ocorreu um erro ao copiar o arquivo PDF: {str(e)}")

# Exemplo de uso da função
#if __name__ == "__main__":
#     # Caminho do arquivo PDF de origem
#     origem_pdf = "XPTO.pdf"
#     
#     # Diretório de destino
#     destino_diretorio = "DirTrein"
#     
#     # Tipo a ser acrescentado ao nome do arquivo
#     especializada = "PPE"
    
    # Chama a função para copiar o PDF para o diretório específico com o tipo acrescentado

@app.route('/ajustar', methods=['POST'])
def resposta3():  
    caminho = r"C:CAMINHO DO PDF"  
    caminho_dir = r"CAMINHO DO DIRETORIO DIRTREIN"
    copiar_pdf_para_diretorio(caminho, caminho_dir, "PPE")
    print('Sucesso')
    return jsonify({'message': 'Model trained successfully!', 'classification_report': copiar_pdf_para_diretorio(caminho, caminho_dir, "INPUT PARA ESPECIALIZADA")})

    #origem pdf= input de pdf, input de string especializada
    #'PAS'
    # 'PDA' 
    # 'PPE' 
    # 'PSE' 
    # 'PTR' 
    # 'PUMA'
    # 'PTA' 



if __name__ == '__main__':
  app.run(host='SEU IP', debug=True)