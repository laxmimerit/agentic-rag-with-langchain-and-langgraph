from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
import json

# Step 1: Define the Pydantic model for structured output
class QueryFilter(BaseModel):
    """Structured output model for extracting company name and other fields."""
    company: Optional[str] = Field(description="The name of the company referred to in the query.")
    # Add more fields here as needed, e.g., industry, location, etc.

query_parser = PydanticOutputParser(pydantic_object=QueryFilter)

# Step 2: Initialize the LLM
model = "qwen2.5"
llm = ChatOllama(model=model, base_url="http://localhost:11434")

# Step 3: Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Extract the relevant information from the following query. Focus on identifying the company name and any other specified fields.

    Query: {query}

    Return the extracted information in the following JSON format:
    {{
        "company": "name of the company"
    }}

    If a field is not mentioned in the query, set it to null. Output MUST be in lowercase only.
    You must return valid JSON with only the following fields: company.
    Do NOT include trailing commas in the JSON.
    """
)

# Step 4: Define a method to run the chain
def get_query_filter(query: str) -> Optional[dict]:
    chain = (
        {"query": RunnablePassthrough()}  # Pass the query directly
        | prompt_template  # Format the query into the prompt
        | llm  # Pass the prompt to the LLM
    )
    
    # Get the raw output from the LLM
    raw_output = chain.invoke(query)
    
    # Clean the output to ensure valid JSON
    try:
        # Remove markdown code block syntax if present
        if raw_output.content.startswith("```json"):
            raw_output = raw_output.content.strip("```json").strip("```").strip()
        else:
            raw_output = raw_output.content.strip()
        
        # Parse the JSON string into a dictionary
        result_dict = json.loads(raw_output)
        
        # Remove None or empty fields
        result_dict = {k: v for k, v in result_dict.items() if v not in [None, '', 'none', 'null', 'nil', 'None', 'Null', 'Nil']}
        
        if not result_dict:
            return None
        
        return result_dict
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None

# Step 5: Example usage
query = "Tell me about the latest news from Apple."
result = get_query_filter(query)
print(result)