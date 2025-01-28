import re
import json
from typing import Dict, Any

# Step 3: Create Tool for the Agent
def get_filter(input_str: str) -> dict:
    """
    Extract the query and metadata filters from the input and return them.
    Input format: 'query="your query", filter={"key": "value"}'.
    """

    # Use regex to extract query and filter from the input string
    query_match = re.search(r'query="([^"]+)"', input_str)
    filter_match = re.search(r'filter=(\{.*?\})', input_str)
    
    if not query_match or not filter_match:
        raise ValueError("Invalid input format. Expected: 'query=\"your query\", filter={\"key\": \"value\"}'")
    
    query = query_match.group(1)
    filter_str = filter_match.group(1)
    
    try:
        # Convert filter string to dictionary
        filter_dict = json.loads(filter_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid filter format. Expected a valid JSON dictionary.")
    
    # Return the extracted query and filter dictionary
    return {"query": query, "filter": filter_dict}
