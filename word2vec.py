import streamlit as st
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime
import time, json, os
import numpy as np
import weaviate
#from weaviate.client import Property, DataType
import gensim.downloader as api
# Import Property and DataType classes from the correct module
#from weaviate.schema.properties import Property
#from weaviate.schema import DataType


from sklearn.metrics.pairwise import cosine_similarity

# Function to initialize the WebDriver
def initialize_driver():
    # Set the desired window size & other options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    # Create a new Chrome window
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# Function to navigate to a section
def navigate_to_section(driver, section_id):
    # Open the webpage
    driver.get("https://arxiv.org/")
    time.sleep(4)
    
    # Find the Computer Science section
    computer_science_heading = driver.find_element(By.XPATH, ".//h2[text()='Computer Science']")
    computer_science_list = computer_science_heading.find_element(By.XPATH, "./following-sibling::ul")
    section_tag = computer_science_list.find_element(By.ID, section_id)
    section_tag.click()
    time.sleep(4)

# Function to get the maximum number of entries
def get_max_entries(driver):
    try:
        max_entries_text = driver.find_element(By.XPATH, "//h3").text
        max_entries = int(max_entries_text.split()[-2])
        return max_entries
    except NoSuchElementException:
        print("Max entries not found.")
        return None

# Function to extract documents
def extract_documents(driver, max_entries, section_id):
    documents = {}
    current_page = 1

    # Click on the "Show All" button if it exists
    try:
        show_all_button = driver.find_element(By.XPATH, '//*[@id="dlpage"]/small[2]/a[3]')
        show_all_button.click()
        time.sleep(4)  # Adding a small delay to ensure all items are loaded
    except NoSuchElementException:
        print('not found')
        pass  # If "Show All" button is not found, continue with regular extraction
    
    # Find all elements matching the XPath expressions
    identifier_elements = driver.find_elements(By.XPATH, "//span[@class='list-identifier']/a[1]")
    title_elements = driver.find_elements(By.XPATH, "//div[@class='list-title mathjax']")
    authors_elements = driver.find_elements(By.XPATH, "//div[@class='list-authors']")
    subjects_elements = driver.find_elements(By.XPATH, "//div[@class='list-subjects']")
    pdflink_elements = driver.find_elements(By.XPATH, "//span[@class='list-identifier']/a[2]")
    
    # Extract the data from the elements
    identifiers = [element.text for element in identifier_elements]
    titles = [element.text for element in title_elements]
    authors = [element.text.replace("Authors:", "").strip().replace("\n", ", ") for element in authors_elements]
    subjects = [element.text.replace("Subjects:", "").strip().replace(";", ",") for element in subjects_elements]
    pdflinks = [element.get_attribute("href") for element in pdflink_elements]
    
    # Store the extracted data in a dictionary
    for identifier, title, author, subject, pdflink in zip(identifiers, titles, authors, subjects, pdflinks):
        documents[identifier] = {
            "Title": title,
            "Authors": author,
            "Subjects": subject,
            "Pdflink": pdflink
        }

        # Print the document details in the terminal
        print("Identifier:", identifier)
        print("Title:", title)
        print("Authors:", author)
        print("Subjects:", subject)
        print("Pdflink:", pdflink)
        print()
        
        # Decrement the counter for extracted entries
        max_entries -= 1
        if max_entries == 0:
            return documents
        
    return documents

# Get the current date and time
current_datetime = datetime.now().strftime("%d-%m-%Y")
# Generate the filename
filename = f"arxiv_data_{current_datetime}.json"

# Function to scrape ArXiv data
def scrape_arxiv_data():
    # Initialize the WebDriver
    driver = initialize_driver()

    # Data storage dictionary
    all_documents = {}

    # Sections to scrape
    sections = ["cs.AI", "cs.CV", "cs.LG"]
    #sections = ["cs.AI"]

    for section_id in sections:
        # Navigate to the section
        navigate_to_section(driver, section_id)

        # Get the maximum number of entries
        max_entries = get_max_entries(driver)

        # Extract documents
        documents = extract_documents(driver, max_entries, section_id)

        # Update the all_documents dictionary
        all_documents.update(documents)

    # Close the WebDriver
    driver.quit()

    # Save the scraped data into a JSON file
    with open(filename, "w") as json_file:
        json.dump(all_documents, json_file)

    return all_documents
    
# Load the scraped data from the JSON file
def load_data_from_json(filename):
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    return data

# Function to preprocess the query
def preprocess_query(query):
    # Remove stopwords and convert to lowercase
    stopwords = {"with", "and", "or", "the", "a", "an", "for", "in", "of", "on", "at", "to", "by"}
    query_words = query.lower().split()
    processed_query = [word for word in query_words if word not in stopwords]
    return processed_query

# Function to load ArXiv data from JSON file
def load_arxiv_data():
    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            return json.load(json_file)
    else:
        return None

import re

def import_data_to_weaviate(data, client, word2vec_model):
    for document_id, document_info in data.items():
        title = document_info.get('Title').split()
        title_vector = np.zeros(word2vec_model.vector_size)  # Initialize a zero vector
        count = 0  # Initialize a count to keep track of valid words
        for word in title:
            if word in word2vec_model:
                title_vector += word2vec_model.get_vector(word)
                count += 1
        if count > 0:
            title_vector /= count  # Take the average of word vectors
        else:
            print(f"Warning: No valid words found in the title for document ID {document_id}. Skipping.")
            continue

        info = client.collections.get("arxiv_data")
        info.data.insert(
            properties={
                "Identifier": document_id,
                "Title": document_info.get('Title'),
                "Authors": document_info.get('Authors'),
                "Subjects": document_info.get('Subjects'),
                "Pdflink": document_info.get('Pdflink')
            },
            vector=title_vector.tolist()  # Convert to list for insertion
        )


# Function to encode query using Word2Vec
def encode_query(query, word2vec_model):
    query_vector = np.zeros(word2vec_model.vector_size)  # Initialize an array to store the vector representation of the query
    words = query.lower().split()  # Split the query into words and convert to lowercase
    num_words = 0
    for word in words:
        if word in word2vec_model:
            query_vector += word2vec_model[word]
            num_words += 1
    if num_words > 0:
        query_vector /= num_words  # Take the average of word vectors to get the query vector
    return query_vector

# Function to encode document titles using Word2Vec
def encode_titles(data, word2vec_model):
    title_vectors = {}
    for document_id, document_info in data.items():
        title = document_info.get('Title', '')
        title_vector = np.zeros(word2vec_model.vector_size)  # Initialize an array to store the vector representation of the title
        words = title.lower().split()  # Split the title into words and convert to lowercase
        num_words = 0
        for word in words:
            if word in word2vec_model:
                title_vector += word2vec_model[word]
                num_words += 1
        if num_words > 0:
            title_vector /= num_words  # Take the average of word vectors to get the title vector
        title_vectors[document_id] = title_vector
    return title_vectors

def preprocess_query_sentence(query_sentence):
    # Define stopwords
    stopwords = {"with", "and", "or", "the", "a", "an", "for", "in", "of", "on", "at", "to", "by"}

    # Tokenize the query sentence into words
    query_words = query_sentence.split()

    # Remove stopwords and convert to lowercase
    processed_query = [word.lower() for word in query_words if word.lower() not in stopwords]

    return processed_query

import streamlit as st

def main():
    data = load_arxiv_data()
    if data is None:
        data = scrape_arxiv_data()

    word2vec_model = api.load('word2vec-google-news-300')
    model = word2vec_model

    client = weaviate.connect_to_embedded(
        persistence_data_path="./word2vec"
    )

     # Schema
    # client.collections.create(
    #     "arxiv_data_word2vec",
    #     properties=[
    #         Property(name="Identifier", data_type=DataType.TEXT),
    #         Property(name="Title", data_type=DataType.TEXT),
    #         Property(name="Authors", data_type=DataType.TEXT),
    #         Property(name="Subjects", data_type=DataType.TEXT),
    #         Property(name="Pdflink", data_type=DataType.TEXT)
    #     ]
    # )

    # Import vectorized data into Weaviate
    # import_data_to_weaviate(data, client, model)

    st.title("Arxiv Search Engine")
    query = st.text_input("Enter your query:")
    
    if st.button("Search"):
        query_vector = np.zeros(word2vec_model.vector_size)
        query_words = query.split()
        count = 0
        for word in query_words:
            if word in word2vec_model:
                query_vector += word2vec_model.get_vector(word)
                count += 1
        if count > 0:
            query_vector /= count
        else:
            st.warning("Warning: No valid words found in the query.")

        inf = client.collections.get('arxiv_data')

        import weaviate.classes as wvc

        response = inf.query.near_vector(
            near_vector=query_vector.tolist(),
            return_metadata=wvc.query.MetadataQuery(distance=True),
            limit=3,
        )

        st.write("Search Results:")
        for o in response.objects:
            st.write("\nTitle:", o.properties['title'])
            st.write("Identifier:", o.properties['identifier'])
            st.write("Authors:", o.properties['authors'])
            st.write("Subjects:", o.properties['subjects'])
            st.write("PDF Link:", o.properties['pdflink'])
            st.write("Distance:", o.metadata.distance)

if __name__ == "__main__":
    main()