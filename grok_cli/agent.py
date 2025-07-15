from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from composio_langchain import ComposioToolSet, App
import os
from pathlib import Path

class GrokAgent:
    def __init__(self, api_key, model="grok-4-0709", base_url="https://api.x.ai/v1"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Set working directory restriction
        self.working_dir = Path(os.getcwd()).resolve()
        
        self.composio_toolset = ComposioToolSet()
        self.tools = self._get_restricted_tools()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with access to file tools. Use the tools when needed to help the user."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,  
            max_iterations=10
        )
    
    def _is_path_allowed(self, file_path):
        """Check if file path is within working directory"""
        try:
            resolved_path = Path(file_path).resolve()
            return resolved_path.is_relative_to(self.working_dir)
        except (ValueError, OSError):
            return False
    
    def _get_restricted_tools(self):
        """Get file tools with working directory restrictions"""
        original_tools = self.composio_toolset.get_tools(apps=[App.FILETOOL])
        
        # Wrap tools with path validation
        restricted_tools = []
        for tool in original_tools:
            restricted_tools.append(self._wrap_tool_with_validation(tool))
        
        return restricted_tools
    
    def _wrap_tool_with_validation(self, tool):
        """Wrap a tool to validate file paths"""
        original_func = tool.func
        
        def validated_func(*args, **kwargs):
            # Check for file path arguments and validate them
            for arg in args:
                if isinstance(arg, str) and ('/' in arg or '\\' in arg):
                    if not self._is_path_allowed(arg):
                        return f"Error: Access denied. File operations are restricted to the working directory: {self.working_dir}"
            
            for value in kwargs.values():
                if isinstance(value, str) and ('/' in value or '\\' in value):
                    if not self._is_path_allowed(value):
                        return f"Error: Access denied. File operations are restricted to the working directory: {self.working_dir}"
            
            return original_func(*args, **kwargs)
        
        tool.func = validated_func
        return tool
    
    def chat(self, user_message):
        """Chat with the agent using LangChain's built-in tool calling"""
        try:
            response = self.agent_executor.invoke({"input": user_message})
            output = response.get("output", "Sorry, I couldn't generate a response.")
            print(output)  
            return output
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                error_msg = "Rate limit exceeded. Please try again later."
                print(error_msg)
                return error_msg
            error_msg = f"An error occurred: {str(e)}"
            print(error_msg)
            return error_msg