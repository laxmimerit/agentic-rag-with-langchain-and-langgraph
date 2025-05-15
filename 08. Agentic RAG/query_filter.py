from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# Step 1: Initialize the LLM
# # model = "llama3.2:3b"
model = "qwen2.5"
llm = ChatOllama(model=model, base_url="http://localhost:11434")

# Step 2: Define the prompt template with examples
prompt_template = ChatPromptTemplate.from_template(
    """
    Extract the company name from the query. Return only the company name in lowercase. If no company is mentioned, return "none".

    you must output company names from the following list only
    - google
    - facebook
    - amazon
    - none

    Examples:
    Query: Tell me about the revenue from Apple.
    Company: apple

    Query: What is the total revenue of Google in 2024?
    Company: google

    Query: How is the weather today?
    Company: none

    Query: {query}
    Company:
    """
)

# Step 3: Define a method to run the chain
def get_company_name(query: str) -> str:
    chain = (
        {"query": RunnablePassthrough()}  # Pass the query directly
        | prompt_template  # Format the query into the prompt
        | llm  # Pass the prompt to the LLM
    )
    
    # Get the raw output from the LLM
    raw_output = chain.invoke(query)
    
    # Clean the output to ensure it's in lowercase and stripped of extra spaces
    company_name = raw_output.content.strip().lower()
    
    # Return "none" if no company is found
    if company_name in ["", "none", "null", "nil"]:
        return "none"
    
    return company_name

# # Step 4: Example usage
# query = "Tell me about the latest news from Microsoft."
# result = get_company_name(query)
# print(result)  # Output: microsoft

# query = "What is the weather today?"
# result = get_company_name(query)
# print(result)  # Output: none