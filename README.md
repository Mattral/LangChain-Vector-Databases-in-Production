# LangChain-Vector-Databases-in-Production

## Building Applications Powered by LLMs with LangChain (Introduction)

LangChain is designed to assist developers in building end-to-end applications using language models. It offers an array of tools, components, and interfaces that simplify the process of creating applications powered by large language models and chat models. LangChain streamlines managing interactions with LLMs, chaining together multiple components, and integrating additional resources, such as APIs and databases. Having gained a foundational understanding of the library in previous lesson, let's now explore various examples of utilizing prompts to accomplish multiple tasks.

## Prompt use case:
A key feature of LangChain is its support for prompts, which encompasses prompt management, prompt optimization, and a generic interface for all LLMs. The framework also provides common utilities for working with LLMs.

ChatPromptTemplate is used to create a structured conversation with the AI model, making it easier to manage the flow and content of the conversation. In LangChain, message prompt templates are used to construct and work with prompts, allowing us to exploit the underlying chat model's potential fully.

System and Human prompts differ in their roles and purposes when interacting with chat models. SystemMessagePromptTemplate provides initial instructions, context, or data for the AI model, while HumanMessagePromptTemplate are messages from the user that the AI model responds to.

To illustrate it, let’s create a chat-based assistant that helps users find information about movies. Ensure your OpenAI key is stored in environment variables using the “OPENAI_API_KEY” name. Remember to install the required packages with the following command: pip install langchain deeplake openai tiktoken.


```
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

response = chat(chat_prompt.format_prompt(movie_title="Inception").to_messages())

print(response.content)
```

```
Inception is a 2010 science fiction action film directed by Christopher Nolan. The film stars Leonardo DiCaprio, Ken Watanabe, Joseph Gordon-Levitt, Ellen Page, Tom Hardy, Dileep Rao, Cillian Murphy, Tom Berenger, and Michael Caine. The plot follows a professional thief who steals information by infiltrating the subconscious of his targets. He is offered a chance to have his criminal history erased as payment for the implantation of another person's idea into a target's subconscious. The film was a critical and commercial success, grossing over $829 million worldwide and receiving numerous accolades, including four Academy Awards.
```

Using the to_messages object in LangChain allows you to convert the formatted value of a chat prompt template into a list of message objects. This is useful when working with chat models, as it provides a structured way to manage the conversation and ensures that the chat model can understand the context and roles of the messages.

### Summarization chain example:
LangChain prompts can be found in various use cases, such as summarization or question-answering chains. For example, when creating a summarization chain, LangChain enables interaction with an external data source to fetch data for use in the generation step. This could involve summarizing a lengthy piece of text or answering questions using specific data sources.

The following code will initialize the language model using OpenAI class with a temperature of 0 - because we want deterministic output.  The load_summarize_chain function accepts an instance of the language model and returns a pre-built summarization chain. Lastly, the PyPDFLoader class is responsible for loading PDF files and converting them into a format suitable for processing by LangChain. 

It is important to note that you need to install the pypdf package to run the following code. Although it is highly recommended to install the latest versions of this package, the codes have been tested on version 3.10.0. Please refer to course introduction lesson for more information on installing packages. 

```
# Import necessary modules
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Initialize language model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="path/to/your/pdf/file.pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])
```

```
This document provides a summary of useful Linux commands for starting and stopping, accessing and mounting file systems, finding files and text within files, the X Window System, moving, copying, deleting and viewing files, installing software, user administration, little known tips and tricks, configuration files and what they do, file permissions, X shortcuts, printing, and a link to an official Linux pocket protector.
```

In this example, the code uses the default summarization chain provided by the load_summarize_chain function. However, you can customize the summarization process by providing prompt templates.

Let’s recap:  OpenAI is initialized with a temperature of 0 for focused and deterministic language model generation. The load_summarize_chain function loads a summarization chain, and PyPDFLoader fetches PDF data, which is loaded as a string input for the summarization chain, generating a summary of the text.

### QA chain example:
We can also use LangChain to manage prompts for asking general questions from the LLMs. These models are proficient in addressing fundamental inquiries. Nevertheless, it is crucial to remain mindful of the potential issue of hallucinations, where the models may generate non-factual information. To address this concern, we will later introduce the Retrieval chain as a means to overcome this problem.

```
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)
```

We define a custom prompt template by creating an instance of the PromptTemplate class. The template string contains a placeholder {question} for the input question, followed by a newline character and the "Answer:" label.  The input_variables argument is set to the list of available placeholders in the prompt (like a question in this case) to indicate the name of the variable that the chain will replace in the template.run() method.

We then instantiate an OpenAI model named gpt-3.5-turbowith a temperature of 0. The OpenAI class is used to create the instance, and the model_name and temperature arguments are provided. Finally, we create a question-answering chain using the LLMChain class. 

The class constructor takes two arguments: llm, which is the instantiated OpenAI model, and prompt, which is the custom prompt template we defined earlier. 

By following these steps, we can process input questions effectively with the custom question-answering, generating appropriate answers using the OpenAI model and the custom prompt template.

```
chain.run("what is the meaning of life?")
```
 
```
'The meaning of life is subjective and can vary from person to person. For some, it may be to find happiness and fulfillment, while for others it may be to make a difference in the world. Ultimately, the meaning of life is up to each individual to decide.’
```
This example demonstrates how LangChain simplifies the integration of LLMs with custom data sources and prompt templates for question-answering applications. To build more advanced NLP applications, you can further extend this example to include other components, such as data-augmented generation, agents, or memory features.

LangChain's support for chain sequences also allows developers to create more complex applications with multiple calls to LLMs or other utilities. These chains can serve various purposes: personal assistants, chatbots, querying tabular data, interacting with APIs, extraction, evaluation, and summarization.
