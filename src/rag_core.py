from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama

def create_rag_chain(retriever):
    """
    Creates a conversational RAG chain from a given retriever.
    """
    # 1. Setup the LLM
    llm = ChatOllama(model="llama3")

    # 2. Setup the Prompt for rephrasing the question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 3. Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 4. Setup the Prompt for answering the question
    qa_system_prompt = (
    "You are a helpful and knowledgeable study assistant. Use the following context "
    "from the course material to answer the question. Provide a thorough, detailed, "
    "and well-explained response that covers all relevant aspects of the topic. "
    "If you don't know the answer based on the context, just say that you don't know—"
    "do not make up an answer.\n\n"
    
    "### Example 1\n"
    "Context:\n"
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize "
    "foods with the help of chlorophyll. It typically involves the green pigment chlorophyll and generates oxygen as a by-product.\n"
    "Question:\n"
    "What is photosynthesis?\n"
    "Answer:\n"
    "Photosynthesis is a biological process that enables green plants, algae, and some bacteria to convert light energy, usually from the sun, into chemical energy in the form of glucose. "
    "This process takes place in the chloroplasts of plant cells, where the green pigment chlorophyll captures light energy. The light energy drives the conversion of carbon dioxide from the air and water from the soil into glucose and oxygen. "
    "The overall chemical reaction can be summarized as: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. This process is essential for life on Earth because it forms the base of the food chain and produces the oxygen necessary for most organisms to survive.\n\n"

    "### Example 2\n"
    "Context:\n"
    "In economics, opportunity cost refers to the value of the next best alternative that is forgone when making a decision. It represents the benefits an individual, investor, or business misses out on when choosing one alternative over another.\n"
    "Question:\n"
    "What is opportunity cost?\n"
    "Answer:\n"
    "Opportunity cost is an important economic concept that refers to the value of the next best alternative that must be given up when a choice is made. It reflects the trade-offs inherent in every decision, where selecting one option means forgoing another. "
    "For example, if a student chooses to spend time studying for an exam instead of working a part-time job, the opportunity cost is the wage they would have earned during that time. "
    "Understanding opportunity cost helps individuals and businesses make more informed decisions by considering what they are sacrificing in order to pursue a specific option.\n\n"

    "### Now use the following context to answer the next question:\n\n"
    "{context}"
)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 5. Create the chain that combines documents
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. Create the final retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain