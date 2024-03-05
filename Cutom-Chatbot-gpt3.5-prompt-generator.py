import os # The line import os imports the os module in Python. This module provides functions for interacting with the operating system, such as accessing environment variables, working with files and directories, and executing system commands.
import sys # The line import sys imports the sys module in Python. This module provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter.

import openai #The line import openai imports the openai module in Python. This module provides access to the OpenAI API, allowing developers to interact with OpenAI's services, such as natural language processing models like GPT (Generative Pre-trained Transformer).
from langchain.chains import ConversationalRetrievalChain, RetrievalQA #ConversationalRetrievalChain: This class likely represents a component or system for managing conversational interactions, possibly integrating natural language processing models and retrieval mechanisms to facilitate conversational query-response interactions. RetrievalQA: This class might represent a system or module focused on retrieval-based question answering, where queries are answered based on retrieved information from a knowledge base or dataset.
from langchain.chat_models import ChatOpenAI #The line from langchain.chat_models import ChatOpenAI imports the ChatOpenAI class from the chat_models module within the langchain package in Python. ChatOpenAI: This class likely represents a model or component designed for conducting conversational interactions using OpenAI's language models. It may provide methods for generating responses to user queries or prompts in a conversational manner, leveraging the capabilities of OpenAI's models.
from langchain.document_loaders import DirectoryLoader, TextLoader # allows the Python script to utilize the DirectoryLoader and TextLoader classes for loading documents and textual data, respectively, from within the langchain package. These loaders may be used to prepare data for further processing or analysis within the script.
from langchain.embeddings import OpenAIEmbeddings #allows the Python script to access and utilize the OpenAIEmbeddings class from the langchain package, potentially enabling the generation of embeddings for textual data using OpenAI's models or techniques. These embeddings can then be used for tasks such as similarity comparison, clustering, or classification.
from langchain.indexes import VectorstoreIndexCreator #The line `from langchain.indexes import VectorstoreIndexCreator` enables the script to access functionality related to creating indexes, particularly the `VectorstoreIndexCreator` class, within the `indexes` module of the `langchain` package. This class likely provides methods and utilities for creating index structures optimized for vector-based data, which can be utilized for efficient information retrieval and similarity search tasks.
from langchain.indexes.vectorstore import VectorStoreIndexWrapper #The line `from langchain.indexes.vectorstore import VectorStoreIndexWrapper` allows the script to access the `VectorStoreIndexWrapper` class from the `vectorstore` module within the `indexes` package of the `langchain` library. This class likely provides a wrapper or interface for interacting with a vector store index, offering functionalities for querying and retrieving data efficiently based on vector representations.
from langchain.llms import OpenAI #The line `from langchain.llms import OpenAI` imports the `OpenAI` class from the `llms` module within the `langchain` package. This class presumably represents an interface or utility for interacting with OpenAI's language models (LLMs) within the context of the `langchain` library. It likely provides methods for initializing, configuring, and utilizing OpenAI's language models for various natural language processing tasks.
from langchain.vectorstores import Chroma #The line `from langchain.vectorstores import Chroma` imports the `Chroma` class from the `vectorstores` module within the `langchain` package. This class likely represents a vector store implementation named "Chroma" within the `langchain` library. Vector stores are data structures optimized for storing and retrieving vector representations of data efficiently. The `Chroma` class may provide methods and utilities for managing and querying vector data stored in this specific format.

#import constants

os.environ["OPENAI_API_KEY"] = "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" #sets the OpenAI API key as an environment variable in the current Python session. This API key is assigned the value "", allowing the script to authenticate and access OpenAI's services, such as their language models and APIs, using this key.

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False #The line `PERSIST = False` sets the value of the variable `PERSIST` to `False`. This variable is likely used as a flag or configuration parameter in the script. In this context, setting `PERSIST` to `False` indicates that a certain functionality, such as saving a model to disk and reusing it for repeated queries, is disabled. Depending on its usage elsewhere in the script, this variable may control various behaviors or options.

query = None #The line `query = None` initializes the variable `query` with the value `None`. This variable is commonly used to store a user's query or input in the script. By initializing it with `None`, it indicates that there is currently no query provided. Later in the script, the value of `query` may be updated based on user input or other conditions.

#sys.argv is a list in Python that contains the command-line arguments passed to the script.
#sys.argv[0] is always the name of the script itself, while subsequent elements contain the command-line arguments provided.
#len(sys.argv) returns the number of elements in sys.argv, which includes at least one (the script name).
#So, len(sys.argv) > 1 checks if there are additional arguments passed (other than the script name).
#If len(sys.argv) > 1, it means there are command-line arguments provided.
#query = sys.argv[1] assigns the value of the first command-line argument (after the script name) to the variable query.
if len(sys.argv) > 1: 
  query = sys.argv[1]

# If the variable PERSIST is True.
# If a directory named "persist" exists in the file system.
# PERSIST is a flag that likely determines whether to save a model to disk and reuse it for repeated queries on the same data.
# os.path.exists("persist") checks if a directory named "persist" exists in the file system.
# If both conditions are true:
# It prints "Reusing index..." to indicate that the index is being reused.
# It initializes a Chroma vector store named vectorstore, specifying the directory "persist" as the persist directory and using OpenAIEmbeddings() as the embedding function.
# It wraps the vectorstore in a VectorStoreIndexWrapper named index.
if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)

# This else block handles the scenario where either the PERSIST flag is False or the "persist" directory does not exist. Here's what it does:
# It initializes a TextLoader named loader, specifying the file path "/Users/praveen18kumar/Downloads/Info.txt". This loader is used to load text data from a single file.
# It checks if the PERSIST flag is True. If it is:
# It initializes an index using VectorstoreIndexCreator, specifying the persist directory as "persist", and loads data from the loader.
# If the PERSIST flag is False or not set:
# It initializes an index using VectorstoreIndexCreator without specifying a persist directory, and loads data from the loader.
else:
  loader = TextLoader("/Users/praveen18kumar/Downloads/Info.txt") # Use this line if you only need data.txt
  #loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

# This line of code initializes a ConversationalRetrievalChain object using components from the langchain library. Here's a breakdown:
# ConversationalRetrievalChain.from_llm: This method is used to create a conversational retrieval chain object from a language model (LLM) and a retriever.
# llm=ChatOpenAI(model="gpt-3.5-turbo"): This specifies the language model to be used in the conversational chain. It initializes a ChatOpenAI model with the GPT-3.5 Turbo variant.
# retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}): This specifies the retriever component of the chain. It converts the vector store index (index) to a retriever using the .as_retriever() method and specifies search parameters using the search_kwargs argument, where {"k": 1} indicates retrieving only the top-most relevant result.
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# This code segment implements a loop for handling a conversational interface. Here's what it does:
# It initializes an empty list chat_history to keep track of the conversation history.
# Enters an infinite loop with while True.
# Checks if query is not provided. If query is None, it prompts the user for input using input("Prompt: "). The user's input is assigned to the query variable.
# Checks if the user input matches any of the exit commands ('quit', 'q', 'exit'). If so, the script exits using sys.exit().
# Passes the user's query to the conversational retrieval chain (chain) to get a response. It includes the current query and the chat_history in the input dictionary.
# Prints the response obtained from the conversational retrieval chain.
# Appends the question-answer pair (query-response) to the chat_history.
# Resets query to None for the next iteration, so the loop will prompt the user for input again.
chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None
