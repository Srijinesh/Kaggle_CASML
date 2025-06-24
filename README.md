# Kaggle_CASML:
This project is a part of a GEN AI Kaggle Hackathon for Retrieval Augmented Generation Based Question Answering. The project was achieved by chunking a 750 page pdf textbook on Psychology into Langchain documents and subsequently, chunks which are stored in the FAISS vector data store. Embeddings are done with Huggingface MiniLLM sentence transformer family embedder. Mistral 7B LLM is used for generating answers from the nearest most accurate context extracted through langchain pipelines.
Test questions are stored in queries.json, and pdf parsing and data cleaning code is in the Data folder.
Embedding, chunking and text metadata are studied and performed in the chunking_rag_pipeline notebook in the Chunking RAG Modelling folder.
Testing with the Mistral 7B llm is done in the mistral_7b_testing notebook in the Chunking RAG Modelling folder.


#Testing:
For testing the model and pipeline, run the mistrak_7b_testing notebook, all cells in sequential fashion, with the my_vector_index folder in the same location as the running notebook.

BLEU score results from the competition were achieved at a value of 0.277 from a value of 0.11.


Kaggle Submission Link: https://www.kaggle.com/competitions/casml-generative-ai-hackathon/submissions#
