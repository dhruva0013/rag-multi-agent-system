import yaml
from crewai import Crew, Process
from rag_toolkit import QueryBuilderAgent, RetrieverAgent, ReviewerAgent, DeciderAgent

def load_crew_config(config_path: str) -> dict:
    """Load crew configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """Main function to orchestrate the RAG multi-agent system."""
    # Load configuration
    config = load_crew_config('config/crew.yaml')
    
    # Initialize agents
    query_builder = QueryBuilderAgent()
    retriever = RetrieverAgent()
    reviewer = ReviewerAgent()
    decider = DeciderAgent()
    
    # Create crew
    crew = Crew(
        agents=[query_builder, retriever, reviewer, decider],
        tasks=config['tasks'],
        process=Process.sequential
    )
    
    # Sample user query
    user_query = "What is the capital of France?"
    
    # Run the crew
    result = crew.kickoff(inputs={'query': user_query})
    
    # Output result
    print(result)

if __name__ == "__main__":
    main()