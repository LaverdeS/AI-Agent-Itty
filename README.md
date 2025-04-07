# AI-Agent-Itty
A retrieval-augmented generation (RAG) agent for cooking that leverages [llama_index](https://github.com/jerryjliu/llama_index) for indexing a collection of cookbooks, recipes, and culinary texts. It also integrates web search and external API tools to fetch the latest cooking trends, methods, and recipes on demand.

## Features

- **Document Indexing & Retrieval:**  
  Index a local collection of cooking documents (cookbooks, recipes, culinary history) using llama_index.
  
- **Web Search Integration:**  
  Retrieve up-to-date cooking content, trends, and external resources via web search.

- **API Tools (in_dev):**  
  Integrate with external APIs (e.g., recipe databases, nutritional info services) to enrich responses.

- **Retrieval Augmented Generation (RAG):**  
  Combine indexed documents with generative models to provide comprehensive and context-aware cooking advice.

## WorkFlow

```
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.utils.workflow import draw_all_possible_flows

cooking_agent = AgentWorkflow.from_tools_or_functions(
    system_prompt=system_prompt,
    tools_or_functions=[search_in_cookbook, search_web],
    verbose=False,
)
ctx = Context(cooking_agent)
draw_all_possible_flows(cooking_agent)
```

<p align="center">
<img src="cooking_agent_workflow.png" alt="Workflow" width="500" />
</p>