from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError

from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.docstore.document import Document

from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    AutoGPTOutputThoughtParser,
    BaseAutoGPTOutputParser
)
from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever

import json
import tiktoken





class AutoGPT:
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_name: str,
        ai_role:str,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        feedback_chain: LLMChain,
        summarize_chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
        self_feedback_in_the_loop:bool = True
    ):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.memory = memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.feedback_chain = feedback_chain
        self.summarize_chain = summarize_chain
        self.self_feedback_in_the_loop = self_feedback_in_the_loop
        self.token_estimator:tiktoken.core.Encoding

    @classmethod
    def from_llm_and_tools(
        cls,
        ai_name: str,
        ai_role: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        self_feedback_in_the_loop:bool = True
    ) -> AutoGPT:
        prompt = AutoGPTPrompt(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
        )

        summarize_chain = load_summarize_chain(llm,chain_type='stuff')

        feedback_prompt_text = "Below is a message from me, an AI Agent, assuming the role of {ai_role}.whilst keeping knowledge of my slight limitations as an AI Agent "\
                            "Please evaluate my thought process, reasoning, criticism, plan, and provide a concise paragraph outlining potential improvements."\
                            "Consider adding or removing ideas that do not align with my role and explaining why and do not loop action, prioritizing thoughts based on their significance,"\
                            "or simply refining my overall thought process. your response should be string sentense style, and be considered your memory. \n"\
                            "note: \n your response should be a paragraph instead of using bullet points.\n"\
                            "{feedback_thoughts}'\n"\
                            'memory:{memory}'
        
        
        feedback_prompt = PromptTemplate(
            template = feedback_prompt_text,
            input_variables = ['ai_role','feedback_thoughts','memory']
        )

        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        feedback_chain = LLMChain(llm=llm,prompt=feedback_prompt)
        cls.token_estimator = tiktoken.encoding_for_model(llm.model_name)

        return cls(
            ai_name=ai_name,
            ai_role=ai_role,
            memory=memory,
            chain=chain,
            feedback_chain=feedback_chain,            
            output_parser=output_parser or AutoGPTOutputParser(),
            tools=tools,
            summarize_chain=summarize_chain,
            feedback_tool=human_feedback_tool,
            self_feedback_in_the_loop=self_feedback_in_the_loop,
        )
    
    def _estimate_token(self,text:str) -> str:
        return len(self.token_estimator.encode(text))
    
    def _summarize_text(self,text_list:list) -> str:
        """
         summarize a text to decrease a token size to under 4096 token.
        """
        text = "".join(text_list)
        while (len(text)>4096):
            token_num = self._estimate_token(text)
            summary_text = []
            for i in range(token_num//4096):
                text = text[4096*(i):4096*(i+1)]
                summary_text.append(self.summarize_chain([Document(page_content=text)])['output_text'])
            text = "".join(summary_text)
        return text


    def get_self_feedback(self, assistant_reply: str) -> str:
        """Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.
        Args:
            thoughts (dict): A dictionary containing thought elements like reasoning,
            plan, thoughts, and criticism.
        Returns:
            str: A feedback response generated using the provided thoughts dictionary.
        """
        gpt_thoughts = AutoGPTOutputThoughtParser().parse(assistant_reply)
        if gpt_thoughts.name == 'thoughts':
            thoughts = gpt_thoughts.args['content']
            reasoning = thoughts.get("reasoning", "")
            plan = thoughts.get("plan", "")
            thought = thoughts.get("text", "")
            criticism = thoughts.get("criticism", "")
            feedback_thoughts = thought + reasoning + plan + criticism
            feedback_response =  self.feedback_chain.run(
                {'ai_role':self.ai_role, 'feedback_thoughts':feedback_thoughts,'memory':self.memory}
            )
        else:
            status = gpt_thoughts.name
            content = gpt_thoughts.args['content']
            print(f"error print : {status}:{content}")
            print(f'here is assistant reply : {assistant_reply}')
            feedback_response = None

        return feedback_response

    def run(self, goals: List[str]) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )
        # Interaction Loop
        loop_count = 0
        while True:
            feedback = None

            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.full_message_history,
                memory=self.memory,
                user_input=user_input,
            )
            print('assistant_reply:\n',assistant_reply)
            print('='*40)
            # Print Assistant thoughts
            self.full_message_history.append(HumanMessage(content=user_input))
            self.full_message_history.append(AIMessage(content=assistant_reply))

            action = self.output_parser.parse(assistant_reply)
            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )
            print('here is action result :',result)
            print('='*40)

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} \n"
            )

            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += feedback
            

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))

            # get self feedback comment
            if (self.self_feedback_in_the_loop) and (action.name != FINISH_NAME ):
                feedback = self.get_self_feedback(assistant_reply)

                # replan a action of agent with self feedback
                if feedback :
                    #print(f"self feedback is below \n {feedback}")
                    feedback_memory_to_add = (
                        f"Self Feedback: {feedback}"
                    )
                    print('feedback:\n',feedback)
                    print('='*40)
                    self.memory.add_documents([Document(page_content=feedback_memory_to_add)])
                    self.full_message_history.append(AIMessage(content=feedback))

            if loop_count > 50:
                print('over 50 loops. so,I will stop this loop')
                break