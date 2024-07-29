This notebook is a demonstration of how to perform RAG (Retrieval-Augmented Generation)
training on an open-source LLM model.

When working with an LLM, a common problem is planning for the LLM to answer specific
questions that it won't have the necessary information to answer.  LLMs are very good at 
answering generalized questions that will be related to general understanding of their language
(e.g. "Is happy an emotion?").  However, if you were creating a chatbot for your local 
restaurant's website, the LLM won't know what the store's hours are, or what items are included 
on the menu.  

A way to solve this problem is through RAG.  Using this approach, you can provide an LLM
with additional documentation that it can use to answer questions, for example HTML from 
your restaurant's website that will have information about the menu or hours.

In this example, I provide the open-source LLM from langchain with the full text of the
science fiction novel Ender's Game.  Using this text, the LLM should be able to answer basic
questions about the book that a generalized knowledge of English wouldn't provide, such as
"Who is Ender's sister?".


There are two different notebooks.  The first uses OpenAI word embeddings to find relevant pieces
of text.  However, this will require opening an account with OpenAI and spending money to have
credits available for making queries.  The second uses word embeddings from the python 
package spaCy, which is free, but will likely result in worse answers (the word vectors in
spaCy are ~300 dimentions, whereas for OpenAI they are ~1,500).

The general approach is to take the text from Ender's Game and split it into smaller, more
manageable chunks of 500 characters.  These chunks are converted into vector embeddings, then
saved to a chroma database where the vector embeddings are the keys.  We then input a prompt
(e.g. "Who is Ender's sister?").  The prompt is embedded as well, and the chroma database is
searched for which sections of the text have the most similar embeddings.  These sections are
then analyzed by the LLM to answer the input prompt.  The selected sections, as well as the
answer to the prompt, are then output, in order to show where the LLM got the information
necessary to answer the questions.

This notebook should serve as a starting guide for individuals looking to implement RAG
into their own AI and LLM problems.