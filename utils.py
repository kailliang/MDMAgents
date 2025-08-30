import os
from dotenv import load_dotenv
import json
import random
import re
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from openai import OpenAI
import requests
import time
import sys
import unicodedata # For normalization
import traceback   # For full traceback if other errors occur

# Load environment variables from .env file
load_dotenv()

# Debug/Visualization settings
SHOW_INTERACTION_TABLE = False  # Set to True to display agent interaction table in intermediate mode
SHOW_AGENT_INTERACTIONS = True  # Set to True to display agent participation decisions and interactions

# Global token usage tracking
GLOBAL_TOKEN_USAGE = {
    'total_input_tokens': 0,
    'total_output_tokens': 0,
    'sample_usage': []
}

def add_to_global_usage(input_tokens, output_tokens, sample_id=None):
    """Add token usage to global tracking"""
    GLOBAL_TOKEN_USAGE['total_input_tokens'] += input_tokens
    GLOBAL_TOKEN_USAGE['total_output_tokens'] += output_tokens
    if sample_id is not None:
        GLOBAL_TOKEN_USAGE['sample_usage'].append({
            'sample_id': sample_id,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        })

def get_global_token_usage():
    """Get global token usage statistics"""
    return GLOBAL_TOKEN_USAGE.copy()

def reset_global_token_usage():
    """Reset global token usage counters"""
    GLOBAL_TOKEN_USAGE['total_input_tokens'] = 0
    GLOBAL_TOKEN_USAGE['total_output_tokens'] = 0
    GLOBAL_TOKEN_USAGE['sample_usage'] = []

class Agent:

    def __init__(self, instruction, role, examplers=None, model_info='gemini-2.5-flash-lite-preview-06-17', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        
        # Initialize token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if self.model_info in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
            if 'genai_api_key' in os.environ:
                genai.configure(api_key=os.environ['genai_api_key'])
            else:
                raise ValueError("Gemini API key not configured. Set 'genai_api_key' in .env file or environment variables.")
            self.model = genai.GenerativeModel(self.model_info)
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-4o-mini', 'gpt-4.1-mini']:

            api_key = os.environ.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key not found. Set 'openai_api_key' environment variable.")
            
            self.client = OpenAI(
                api_key=api_key, # API key is now clean
            )
            
            current_instruction_content = str(self.instruction)

            self.messages = [
                {"role": "system", "content": current_instruction_content},
            ]

            if examplers is not None:
                for exampler in examplers:
                    question = str(exampler.get('question', ''))
                    answer = str(exampler.get('answer', ''))
                    reason = str(exampler.get('reason', ''))

                    self.messages.append({"role": "user", "content": question})
                    self.messages.append({"role": "assistant", "content": answer + "\n\n" + reason})
        else:
            raise ValueError(f"Unsupported model_info: {self.model_info}")

    def _clean_problematic_unicode(self, text_content):
        # This function is still useful for cleaning message content before sending to LLMs,
        # especially if they are sensitive to certain Unicode characters or if you want to
        # normalize input. However, it wasn't the cause of the API key header issue.
        if not isinstance(text_content, str):
            if text_content is None:
                return ""
            try:
                text_content = str(text_content)
            except Exception:
                return ""

        try:
            normalized_text = unicodedata.normalize('NFKC', text_content)
        except TypeError:
            normalized_text = text_content
        
        normalized_text = normalized_text.replace('\u201c', '"').replace('\u201d', '"')
        normalized_text = normalized_text.replace('\u2018', "'").replace('\u2019', "'")
        normalized_text = normalized_text.replace('\u2013', '-').replace('\u2014', '--')

        ascii_bytes = normalized_text.encode('ascii', errors='replace')
        cleaned_string = ascii_bytes.decode('ascii')
        
        return cleaned_string
    
    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
            for _ in range(10):
                try:
                    # Gemini expects UTF-8 strings for messages.
                    response = self._chat.send_message(str(message))
                    
                    # Track token usage for Gemini
                    if hasattr(response, 'usage_metadata'):
                        if hasattr(response.usage_metadata, 'prompt_token_count'):
                            self.total_input_tokens += response.usage_metadata.prompt_token_count
                        if hasattr(response.usage_metadata, 'candidates_token_count'):
                            self.total_output_tokens += response.usage_metadata.candidates_token_count
                    
                    return response.text
                except Exception as e:
                    # Safe cprint for error messages
                    error_str = f"Error communicating with Gemini: {e}"
                    safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                    cprint(safe_error_str, "red")
                    time.sleep(1) 
            return "Error: Failed to get response from Gemini after multiple retries."

        elif self.model_info in ['gpt-4o-mini', 'gpt-4.1-mini']:
            # OpenAI also expects UTF-8 strings for message content.
            current_user_message_content = str(message)
            # cleaned_user_message_content = self._clean_problematic_unicode(current_user_message_content) # Potentially remove if not needed

            # self.messages contains original (or lightly cleaned) history
            api_call_messages = [msg.copy() for msg in self.messages]
            # Use original user message content if aggressive cleaning isn't needed
            api_call_messages.append({"role": "user", "content": current_user_message_content})
            
            # model_name = "gpt-4o-mini"

            try:
                response = self.client.chat.completions.create(
                    model=self.model_info,
                    messages=api_call_messages,
                    temperature=0.7
                )
                
                # Track token usage for OpenAI
                if hasattr(response, 'usage'):
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                raw_response_content = response.choices[0].message.content
                # cleaned_response_content = self._clean_problematic_unicode(raw_response_content) # Potentially remove

                # Add original user message and original assistant response to history
                self.messages.append({"role": "user", "content": current_user_message_content})
                self.messages.append({"role": "assistant", "content": raw_response_content})
                return raw_response_content

            except Exception as e: # Catch general exceptions; specific UnicodeEncodeError on headers is now fixed.
                error_str = f"Error communicating with OpenAI: {e}"
                safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                cprint(safe_error_str, "red")
                cprint("FULL TRACEBACK for OpenAI Exception:", "red", force_color=True)
                traceback.print_exc()
                return f"Error: Failed to get response from OpenAI: {str(e)}"
        else:
            raise ValueError(f"Unsupported model_info in chat: {self.model_info}")

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-4o-mini', 'gpt-4.1-mini']:
            current_user_message_content = str(message)
            # cleaned_user_message_content = self._clean_problematic_unicode(current_user_message_content) # Potentially remove

            api_call_messages = [msg.copy() for msg in self.messages]
            api_call_messages.append({"role": "user", "content": current_user_message_content})
            
            responses = {}
            # model_name = "gpt-4o-mini"

            try:
                response = self.client.chat.completions.create(
                    model=self.model_info,
                    messages=api_call_messages,
                    temperature=0.0
                )
                
                # Track token usage for OpenAI
                if hasattr(response, 'usage'):
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                raw_response_content = response.choices[0].message.content
                responses[0.0] = raw_response_content
                return responses

            except Exception as e:
                error_str = f"Error communicating with OpenAI in temp_responses: {e}"
                safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                cprint(safe_error_str, "red")
                cprint("FULL TRACEBACK for OpenAI temp_responses Exception:", "red", force_color=True)
                traceback.print_exc()
                return {0.0: f"Error: Failed to get response from OpenAI: {str(e)}"}
        
        elif self.model_info in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
            try:
                # Configure generation with temperature=0.0 for deterministic responses
                generation_config = genai.GenerationConfig(temperature=0.0)
                response = self._chat.send_message(str(message), generation_config=generation_config)
                
                # Track token usage for Gemini
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        self.total_input_tokens += response.usage_metadata.prompt_token_count
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        self.total_output_tokens += response.usage_metadata.candidates_token_count
                
                return {0.0: response.text}
            except Exception as e:
                error_str = f"Error communicating with Gemini for temp_responses: {e}"
                safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                cprint(safe_error_str, "red")
                return {0.0: "Error: Failed to get response from Gemini."}
        else:
            raise ValueError(f"Unsupported model_info in temp_responses: {self.model_info}")
    
    def get_token_usage(self):
        """Return current token usage for this agent"""
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }
    
    def reset_token_usage(self):
        """Reset token usage counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


class Group:
    def __init__(self, goal, members, question, examplers=None, model_info='gemini-2.5-flash-lite-preview-06-17'):
        self.goal = goal
        self.members = []
        for member_info in members:
            # Group members use gpt-4o-mini
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), 
                           role=member_info['role'], 
                           model_info=model_info)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                if not assist_members:
                    cprint("Warning: Group has no members or no identifiable lead/assistant.", "red")
                    return "Error: Group configuration issue."
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            if assist_members:
                for a_mem in assist_members:
                    delivery_prompt += "\n{}".format(a_mem.role)
            else:
                delivery_prompt += "\nYou are working independently or with a predefined protocol to address the goal."

            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians (if any), or outline your approach. Strictly limit your response with no more than 50 words. \n Question: {}".format(self.question)
            
            try:
                delivery = lead_member.chat(delivery_prompt)
            except Exception as e:
                cprint(f"Error during lead_member chat: {e}", "red")
                if assist_members and lead_member != assist_members[0]:
                    try:
                        delivery = assist_members[0].chat(delivery_prompt)
                    except Exception as e2:
                        cprint(f"Error during fallback assistant chat: {e2}", "red")
                        return "Error: Could not get delivery from group lead or assistants."
                else:
                    return "Error: Could not get delivery from group lead."

            investigations = []
            if assist_members:
                for a_mem in assist_members:
                    investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information. Strictly limit your response with no more than 100 words.".format(self.goal, delivery))
                    investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            if investigations:
                for investigation_item in investigations:
                    gathered_investigation += "[{}]\n{}\n".format(investigation_item[0], investigation_item[1])
            else:
                gathered_investigation = delivery

            # Direct reasoning without few-shot examples
            investigation_prompt = f"""The gathered investigation from your assistant clinicians (or your own initial assessment if working alone) is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided. Limit your response with no more than 100 words.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)
            return response

        elif comm_type == 'external':
            return "External communication not implemented."
        else:
            return "Unknown communication type."

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    count = 0
    
    def get_emoji(index):
        return emojis[index % len(emojis)]
    
    for expert, hierarchy_str in info:
        try:
            expert_name = expert.split('-')[0].split('.')[1].strip()
        except:
            expert_name = expert.split('-')[0].strip()
        
        if hierarchy_str is None:
            hierarchy_str = 'Independent'
        
        if 'independent' not in hierarchy_str.lower():
            hierarchy_parts = hierarchy_str.split(">")
            if len(hierarchy_parts) >= 2:
                parent_name = hierarchy_parts[0].strip()
                child_name = hierarchy_parts[1].strip()

                parent_node_found = False
                for agent_node in agents:
                    if agent_node.name.split("(")[0].strip().lower() == parent_name.strip().lower():
                        child_agent_node = Node("{} ({})".format(child_name, get_emoji(count)), agent_node)
                        agents.append(child_agent_node)
                        parent_node_found = True
                        break
                if not parent_node_found:
                    cprint(f"Warning: Parent '{parent_name}' for child '{child_name}' not found. Adding child to moderator.", "yellow")
                    agent_node = Node("{} ({})".format(expert_name, get_emoji(count)), moderator)
            else:
                # Invalid hierarchy format, treat as independent
                cprint(f"Warning: Invalid hierarchy format '{hierarchy_str}' for expert '{expert_name}'. Treating as independent.", "yellow")
                agent_node = Node("{} ({})".format(expert_name, get_emoji(count)), moderator)
                agents.append(agent_node)
        else:
            agent_node = Node("{} ({})".format(expert_name, get_emoji(count)), moderator)
            agents.append(agent_node)
        count += 1
    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    parsed_info = {
        'group_goal': '',
        'members': []
    }
    if not lines or not lines[0]:
        return parsed_info

    goal_parts = lines[0].split('-', 1)
    if len(goal_parts) > 1:
        parsed_info['group_goal'] = goal_parts[1].strip()
    else:
        parsed_info['group_goal'] = goal_parts[0].strip()
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_parts = line.split(':', 1)
            if len(member_parts) < 2: continue

            member_role_description_str = member_parts[1].split('-', 1)
            
            member_role = member_role_description_str[0].strip()
            member_expertise = member_role_description_str[1].strip() if len(member_role_description_str) > 1 else 'General expertise'
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    return parsed_info

def setup_model(model_name):
    # Normalize model_name to avoid issues with whitespace/case
    original_model_name = model_name
    model_name = str(model_name).strip()
    print(f"[DEBUG] setup_model received model_name: '{original_model_name}' (normalized: '{model_name}')")
    if model_name in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
        if 'genai_api_key' in os.environ:
            genai.configure(api_key=os.environ['genai_api_key'])
            return True
        else:
            cprint("Error: 'genai_api_key' not found for Gemini setup.", "red")
            return False
    elif model_name in ['gpt-4o-mini', 'gpt-4.1-mini']:
        if 'openai_api_key' in os.environ:
            return True
        else:
            cprint("Error: 'openai_api_key' not found for OpenAI setup.", "red")
            return False
    else:
        supported = ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17', 'gpt-4o-mini', 'gpt-4.1-mini']
        raise ValueError(f"Unsupported model for setup: '{original_model_name}'. Supported models: {supported}")

def load_data(dataset):
    test_qa = []
    examplers = []
    base_data_path = os.path.join(os.path.dirname(__file__), 'data')

    test_path = os.path.join(base_data_path, dataset, 'test.jsonl')
    print(f"[DEBUG] Loading test data from: {test_path}")
    try:
        with open(test_path, 'r', encoding='utf-8') as file:
            for line in file:
                test_qa.append(json.loads(line))
    except FileNotFoundError:
        cprint(f"Error: Test data file not found at {test_path}", "red")

    train_path = os.path.join(base_data_path, dataset, 'train.jsonl')
    try:
        with open(train_path, 'r', encoding='utf-8') as file:
            for line in file:
                examplers.append(json.loads(line))
    except FileNotFoundError:
        cprint(f"Error: Train data file (exemplars) not found at {train_path}", "red")
    print(f"[DEBUG] test_qa loaded: {len(test_qa)}")
    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample.get('question', "No question provided in sample."), None

def determine_difficulty(question, difficulty, model_to_use='gemini-2.5-flash-lite-preview-06-17'):
    if difficulty != 'adaptive':
        return difficulty, 0, 0  # Return difficulty with zero token usage for non-adaptive
    
    difficulty_prompt = f"""Analyze the following medical query and determine its complexity level.

Medical Query:
{question}

**Difficulty Levels:**
- **basic**: a single medical agent can output an answer.
- **intermediate**: number of medical experts with different expertise should dicuss and make final decision.
- **advanced**: multiple teams of clinicians from different departments need to collaborate with each other to make final decision.
Provide your assessment in the following JSON format:

{{
  "difficulty": "basic|intermediate|advanced"
}}

**Requirements:**
- Return ONLY the JSON format, no other text
- Difficulty must be exactly one of: basic, intermediate, advanced
"""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and determines the complexity level of medical queries.', role='medical expert', model_info=model_to_use)
    medical_agent.chat('You are a medical expert who conducts initial assessment and determines the complexity level of medical queries.')
    response_dict = medical_agent.temp_responses(difficulty_prompt)
    response = response_dict.get(0.0, "")

    # Get token usage from the difficulty determination agent
    difficulty_agent_usage = medical_agent.get_token_usage()
    difficulty_input_tokens = difficulty_agent_usage['input_tokens']
    difficulty_output_tokens = difficulty_agent_usage['output_tokens']

    # Parse JSON response
    try:
        json_match = re.search(r'\{\s*"difficulty"\s*:\s*"([^"]+)"\s*\}', response, re.DOTALL)
        if json_match:
            determined_difficulty = json_match.group(1).lower().strip()
            
            if determined_difficulty in ['basic', 'intermediate', 'advanced']:
                return determined_difficulty, difficulty_input_tokens, difficulty_output_tokens
            else:
                cprint(f"Warning: Invalid difficulty level '{determined_difficulty}' in JSON response. Defaulting to intermediate.", "yellow")
                return 'intermediate', difficulty_input_tokens, difficulty_output_tokens
        else:
            # Fallback to original text parsing
            if 'basic' in response.lower():
                return 'basic', difficulty_input_tokens, difficulty_output_tokens
            elif 'intermediate' in response.lower():
                return 'intermediate', difficulty_input_tokens, difficulty_output_tokens
            elif 'advanced' in response.lower():
                return 'advanced', difficulty_input_tokens, difficulty_output_tokens
            else:
                cprint(f"Warning: Could not parse difficulty from response: '{response}'. Defaulting to intermediate.", "yellow")
                return 'intermediate', difficulty_input_tokens, difficulty_output_tokens
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Fallback to original text parsing
        cprint(f"Warning: JSON parsing failed for difficulty response. Using fallback text parsing.", "yellow")
        if 'basic' in response.lower():
            return 'basic', difficulty_input_tokens, difficulty_output_tokens
        elif 'intermediate' in response.lower():
            return 'intermediate', difficulty_input_tokens, difficulty_output_tokens
        elif 'advanced' in response.lower():
            return 'advanced', difficulty_input_tokens, difficulty_output_tokens
        else:
            cprint(f"Warning: Could not parse difficulty from response: '{response}'. Defaulting to intermediate.", "red")
            return 'intermediate', difficulty_input_tokens, difficulty_output_tokens

def process_basic_query(question, model_to_use):
    import re
    import json
    
    # Reset token usage for this sample
    sample_input_tokens = 0
    sample_output_tokens = 0
    
    # Step 1: Expert Recruitment - recruit 3 experts with equal weight
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = "You are an experienced medical expert who recruits medical specialists to solve the given medical query."
    
    recruiter_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_to_use)
    recruiter_agent.chat(recruit_prompt)
    
    num_experts_to_recruit = 3
    recruited_text = recruiter_agent.chat(f"""Question: {question}

You need to recruit {num_experts_to_recruit} medical experts with different specialties but equal authority/weight to analyze this question. 

Please return your recruitment plan in JSON format:

{{
  "experts": [
    {{
      "id": 1,
      "role": "Cardiologist",
      "expertise_description": "Specializes in heart and cardiovascular system disorders",
      "hierarchy": "Independent"
    }},
    {{
      "id": 2,
      "role": "Pulmonologist", 
      "expertise_description": "Specializes in respiratory system diseases",
      "hierarchy": "Independent"
    }},
    {{
      "id": 3,
      "role": "Emergency Medicine Physician",
      "expertise_description": "Specializes in acute care and emergency medical situations", 
      "hierarchy": "Independent"
    }}
  ]
}}

All experts should be marked as "Independent" with equal authority. Return ONLY the JSON, no other text.""")

    # Parse JSON response for expert recruitment
    try:
        # Clean JSON response by removing markdown blocks
        cleaned_response = recruited_text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        
        experts_data = json.loads(cleaned_response)
        recruited_experts = experts_data.get('experts', [])
        
        if len(recruited_experts) != num_experts_to_recruit:
            raise ValueError(f"Expected {num_experts_to_recruit} experts, got {len(recruited_experts)}")
            
    except (json.JSONDecodeError, ValueError) as e:
        cprint(f"Warning: Failed to parse JSON recruitment response: {e}. Using default experts.", "yellow")
        recruited_experts = [
            {"id": 1, "role": "General Internal Medicine Physician", "expertise_description": "Specializes in comprehensive adult medical care", "hierarchy": "Independent"},
            {"id": 2, "role": "Emergency Medicine Physician", "expertise_description": "Specializes in acute care and emergency situations", "hierarchy": "Independent"},  
            {"id": 3, "role": "Family Medicine Physician", "expertise_description": "Specializes in primary care across all age groups", "hierarchy": "Independent"}
        ]
    
    # Display recruited experts
    # print("Recruited Experts:")
    # for expert in recruited_experts:
    #     print(f"Expert {expert['id']}: {expert['role']} - {expert['expertise_description']}")
    # print()
    
    # Step 2: Create agent instances and get individual responses
    cprint("[INFO] Step 2. Independent Expert Analysis", 'yellow', attrs=['blink'])
    expert_agents = []
    expert_responses = []
    
    for expert in recruited_experts:
        # Create agent for each expert
        agent = Agent(
            instruction=f"You are a {expert['role']} who {expert['expertise_description'].lower()}. Your job is to analyze medical questions independently.",
            role=expert['role'], 
            examplers=None,
            model_info=model_to_use
        )
        agent.chat(f"You are a {expert['role']} who {expert['expertise_description'].lower()}.")
        expert_agents.append(agent)
        
        # Get response from each expert with structured JSON format
        expert_prompt = f"""You are a {expert['role']}. Analyze the following multiple choice question and provide your response in exactly this JSON format:

{{
  "reasoning": "Your step-by-step medical analysis in no more than 300 words",
  "answer": "X) Example Answer "
}}

**Requirements:**
- Answer must correspond to one of the provided options
- Return ONLY the JSON, no other text

**Question:** {question}
"""
        
        response_dict = agent.temp_responses(expert_prompt, img_path=None)
        raw_response = response_dict.get(0.0, "")
        
        # Parse expert response
        try:
            json_match = re.search(r'\{[^{}]*"reasoning"\s*:[^{}]*"answer"\s*:\s*"[^"]*"[^{}]*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                expert_response = {
                    "expert_id": expert['id'],
                    "role": expert['role'],
                    "reasoning": parsed_json.get("reasoning", "").strip(),
                    "answer": parsed_json.get("answer", "").strip()
                }
            else:
                expert_response = {
                    "expert_id": expert['id'],
                    "role": expert['role'], 
                    "reasoning": "Unable to parse expert response",
                    "answer": "X) Parse error"
                }
        except json.JSONDecodeError:
            expert_response = {
                "expert_id": expert['id'],
                "role": expert['role'],
                "reasoning": "JSON parsing error",
                "answer": "X) JSON error"
            }
        
        expert_responses.append(expert_response)
        
    # Step 3: Arbitrator analysis and final decision
    cprint("[INFO] Step 3. Arbitrator Final Decision", 'yellow', attrs=['blink'])
    
    arbitrator = Agent(
        instruction="You are a medical arbitrator who reviews multiple expert opinions and synthesizes the best final decision.",
        role="Medical Arbitrator",
        model_info=model_to_use
    )
    arbitrator.chat("You are a medical arbitrator who reviews multiple expert opinions and synthesizes the best final decision.")
    
    # Format expert responses for arbitrator
    experts_summary = ""
    for response in expert_responses:
        experts_summary += f"Expert {response['expert_id']} ({response['role']}):\n"
        experts_summary += f"Reasoning: {response['reasoning']}\n"
        experts_summary += f"Answer: {response['answer']}\n\n"
    
    arbitrator_prompt = f"""You are a medical arbitrator. Review the following expert opinions and provide your final decision in JSON format:

{experts_summary}

Question: {question}

Analyze all expert opinions and provide your final decision in exactly this JSON format:

{{
  "analysis": "Your analysis of the expert opinions and rationale for final decision in no more than 300 words",
  "final_answer": "X) Example Answer"
}}

**Requirements:**
- Consider all expert opinions in your analysis
- Final answer must correspond to one of the provided options
- Return ONLY the JSON, no other text
"""
    
    final_response_dict = arbitrator.temp_responses(arbitrator_prompt, img_path=None)
    raw_final_response = final_response_dict.get(0.0, "")
    
    # Parse arbitrator response
    try:
        json_match = re.search(r'\{[^{}]*"analysis"\s*:[^{}]*"final_answer"\s*:\s*"[^"]*"[^{}]*\}', raw_final_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            final_decision_dict = json.loads(json_str)
        else:
            final_decision_dict = {
                "analysis": "Unable to parse arbitrator response",
                "final_answer": "X) Parse error"
            }
    except json.JSONDecodeError:
        final_decision_dict = {
            "analysis": "JSON parsing error in arbitrator response", 
            "final_answer": "X) JSON error"
        }
    
    print(f"Arbitrator Final Decision: {final_decision_dict.get('final_answer', 'Error')}")
    
    # Calculate token usage for this sample
    recruiter_usage = recruiter_agent.get_token_usage()
    sample_input_tokens += recruiter_usage['input_tokens']
    sample_output_tokens += recruiter_usage['output_tokens']
    
    for agent in expert_agents:
        agent_usage = agent.get_token_usage()
        sample_input_tokens += agent_usage['input_tokens']
        sample_output_tokens += agent_usage['output_tokens']
    
    arbitrator_usage = arbitrator.get_token_usage()
    sample_input_tokens += arbitrator_usage['input_tokens']
    sample_output_tokens += arbitrator_usage['output_tokens']
    
    return final_decision_dict, sample_input_tokens, sample_output_tokens


def process_intermediate_query(question, model_to_use):
    # Reset token usage for this sample
    sample_input_tokens = 0
    sample_output_tokens = 0
    
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = "You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."
    
    recruiter_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_to_use)
    recruiter_agent.chat(recruit_prompt)

    num_experts_to_recruit = 3

    recruitment_query = f"""
    Question: {question}

    You can recruit {num_experts_to_recruit} experts in different medical expertise. 
    Considering the medical question and the options for the answer, what kind of experts 
    will you recruit to better make an accurate answer?

    Also, you need to specify the communication structure between experts 
    (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), 
    or indicate if they are independent.

    Please answer strictly in the following JSON format:
    {{
    "experts": [
        {{
        "id": 1,
        "role": "Pediatrician",
        "description": "Specializes in the medical care of infants, children, and adolescents.",
        "hierarchy": "Independent"
        }},
        {{
        "id": 2,
        "role": "Cardiologist",
        "description": "Focuses on the diagnosis and treatment of heart and blood vessel-related conditions.",
        "hierarchy": "Pediatrician > Cardiologist"
        }},
        {{
        "id": 3,
        "role": "Pulmonologist",
        "description": "Specializes in the diagnosis and treatment of respiratory system disorders.",
        "hierarchy": "Independent"
        }}
    ]
    }}

    Do not include any explanations or reasons, just return the JSON structure.
    """
    recruited_text = recruiter_agent.chat(recruitment_query)

    # Parse JSON response for expert recruitment
    try:
        # Clean JSON response by removing markdown blocks
        cleaned_response = recruited_text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        
        experts_data = json.loads(cleaned_response)
        recruited_experts = experts_data.get('experts', [])
        
        if len(recruited_experts) != num_experts_to_recruit:
            raise ValueError(f"Expected {num_experts_to_recruit} experts, got {len(recruited_experts)}")
        
        # Convert to expected format: [(description, hierarchy), ...]
        agents_data_parsed = []
        for expert in recruited_experts:
            role = expert.get('role', 'Unknown Role')
            description = expert.get('expertise_description', 'No description')
            hierarchy = expert.get('hierarchy', 'Independent')
            expert_info = f"{expert.get('id', len(agents_data_parsed)+1)}. {role} - {description}"
            agents_data_parsed.append((expert_info, hierarchy))
            
    except (json.JSONDecodeError, ValueError) as e:
        cprint(f"Warning: Failed to parse JSON recruitment response: {e}. Using fallback parsing.", "yellow")
        # Fallback to original text parsing for robustness
        agents_info_raw = [agent_info.split(" - Hierarchy: ") for agent_info in recruited_text.split('\n') if agent_info.strip()]
        agents_data_parsed = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info_raw]
        
        # If fallback also fails, use default experts
        if len(agents_data_parsed) != num_experts_to_recruit:
            cprint("Warning: Fallback parsing also failed. Using default experts.", "yellow")
            agents_data_parsed = [
                ("1. General Internal Medicine Physician - Specializes in comprehensive adult medical care", "Independent"),
                ("2. Emergency Medicine Physician - Specializes in acute care and emergency situations", "Independent"),  
                ("3. Family Medicine Physician - Specializes in primary care across all age groups", "Independent")
            ]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    if num_experts_to_recruit > len(agent_emoji):
        agent_emoji.extend(['\U0001F9D1'] * (num_experts_to_recruit - len(agent_emoji)))

    hierarchy_agents_nodes = parse_hierarchy(agents_data_parsed, agent_emoji)

    agent_list_str = ""
    for i, agent_tuple in enumerate(agents_data_parsed):
        try:
            agent_desc_parts = str(agent_tuple[0]).split('-', 1)
            role_part = agent_desc_parts[0].split('.', 1)[-1].strip().lower() if '.' in agent_desc_parts[0] else agent_desc_parts[0].strip().lower()
            description = agent_desc_parts[1].strip().lower() if len(agent_desc_parts) > 1 else "No description"
            agent_list_str += f"Agent {i+1}: {role_part} - {description}\n"
        except Exception as e:
            cprint(f"Error parsing agent data for list: {agent_tuple} - {e}", "red")
            agent_list_str += f"Agent {i+1}: Error parsing agent info\n"

    agent_dict = {}
    medical_agents_list = []
    for agent_tuple in agents_data_parsed:
        try:
            agent_desc_parts = str(agent_tuple[0]).split('-', 1)
            agent_role_parsed = agent_desc_parts[0].split('.', 1)[-1].strip().lower() if '.' in agent_desc_parts[0] else agent_desc_parts[0].strip().lower()
            description_parsed = agent_desc_parts[1].strip().lower() if len(agent_desc_parts) > 1 else "No description"
        except Exception as e:
            cprint(f"Error parsing agent data for instantiation: {agent_tuple} - {e}", "red")
            continue
        
        inst_prompt = f"""You are a {agent_role_parsed} who {description_parsed}. Your job is to collaborate with other medical experts in a team."""
        _agent_instance = Agent(instruction=inst_prompt, role=agent_role_parsed, model_info=model_to_use)
        
        _agent_instance.chat(inst_prompt)
        agent_dict[agent_role_parsed] = _agent_instance
        medical_agents_list.append(_agent_instance)

    for idx, agent_tuple in enumerate(agents_data_parsed):
        if SHOW_INTERACTION_TABLE:
            try:
                emoji = agent_emoji[idx % len(agent_emoji)]
                agent_name_part = str(agent_tuple[0]).split('-')[0].strip()
                agent_desc_part = str(agent_tuple[0]).split('-', 1)[1].strip() if '-' in str(agent_tuple[0]) else "N/A"
                print(f"Agent {idx+1} ({emoji} {agent_name_part}): {agent_desc_part}")
            except IndexError:
                 print(f"Agent {idx+1} ({agent_emoji[idx % len(agent_emoji)]}): {agent_tuple[0]}")
            except Exception as e:
                cprint(f"Error printing agent info: {agent_tuple} - {e}", "red")

    # print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    if hierarchy_agents_nodes and SHOW_INTERACTION_TABLE:
        try:
            from pptree import print_tree
            print_tree(hierarchy_agents_nodes[0], horizontal=False)
        except ImportError:
            cprint("pptree not installed or print_tree not found. Skipping hierarchy print.", "yellow")
        except Exception as e:
            cprint(f"Error printing tree: {e}", "red")


    num_rounds = 3 # 这里改成 3
    num_turns = 3 # 这里改成 3
    num_active_agents = len(medical_agents_list)

    interaction_log = {f'Round {r}': {f'Turn {t}': {f'Agent {s}': {f'Agent {trg}': None for trg in range(1, num_active_agents + 1)} for s in range(1, num_active_agents + 1)} for t in range(1, num_turns + 1)} for r in range(1, num_rounds + 1)}

    # cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions_log = {r: {} for r in range(1, num_rounds+1)}
    initial_report_str = ""
    summarizer_agents_list = []  # Track all summarizer agents for token usage
    
    for agent_role_key, agent_instance in agent_dict.items():
        # Direct reasoning without few-shot examples
        opinion = agent_instance.chat(f'''Please return your answer within 50 words to the medical query among the option provided.\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        initial_report_str += f"({agent_role_key.lower()}): {opinion}\n"
        round_opinions_log[1][agent_role_key.lower()] = opinion

    final_answer_map = None
    for r_idx in range(1, num_rounds+1):
        if SHOW_INTERACTION_TABLE:
            print(f"== Round {r_idx} ==")
        round_name_str = f"Round {r_idx}"
        
        summarizer_agent = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model_to_use)
        summarizer_agent.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        summarizer_agents_list.append(summarizer_agent)  # Track for token usage
        
        current_assessment_str = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions_log[r_idx].items())

        num_participated_this_round = 0
        for t_idx in range(num_turns):
            turn_name_str = f"Turn {t_idx + 1}"
            if SHOW_INTERACTION_TABLE:
                print(f"|_{turn_name_str}")

            num_participated_this_turn = 0
            for agent_idx, agent_instance_loop in enumerate(medical_agents_list):
                context_for_participation_decision = current_assessment_str

                participate_decision_response = agent_instance_loop.chat(f"Given the opinions from other medical experts in your team (see below), please indicate whether you want to talk to any expert. Return your response in JSON format with the key 'participate' (true/false) and 'reason' (brief explanation).\n\nOpinions:\n{context_for_participation_decision}\n\nExample response:\n{{\n  \"participate\": true,\n  \"reason\": \"I disagree with the cardiologist's assessment\"\n}}")
                
                # Parse JSON response for participation decision
                try:
                    participate_json = json.loads(participate_decision_response)
                    should_participate = participate_json.get('participate', False)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    # Fallback to text parsing if JSON parsing fails
                    should_participate = 'yes' in participate_decision_response.lower().strip() or 'true' in participate_decision_response.lower().strip()
                
                # Log agent participation decision
                if SHOW_AGENT_INTERACTIONS:
                    agent_emoji_char = agent_emoji[agent_idx % len(agent_emoji)]
                    participation_status = "WANTS TO TALK" if should_participate else "STAYS SILENT"
                    cprint(f"[PARTICIPATION] Agent {agent_idx+1} ({agent_emoji_char} {agent_instance_loop.role}) - {participation_status}", 'cyan')
                    cprint(f"[DECISION] {participate_decision_response}", 'cyan')
                
                if should_participate:                
                    chosen_expert_response = agent_instance_loop.chat(f"Select which expert(s) you want to talk to (1-{num_active_agents}). Return your response in JSON format with the key 'selected_experts' as an array of numbers.\n\n{agent_list_str}\n\nExample response:\n{{\n  \"selected_experts\": [1, 3]\n}}")
                    
                    # Parse JSON response for expert selection
                    try:
                        expert_json = json.loads(chosen_expert_response)
                        chosen_expert_indices = expert_json.get('selected_experts', [])
                        # Ensure all indices are integers
                        chosen_expert_indices = [int(idx) for idx in chosen_expert_indices if str(idx).isdigit()]
                    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
                        # Fallback to original parsing if JSON parsing fails
                        chosen_expert_indices = [int(ce.strip()) for ce in chosen_expert_response.replace('.', ',').split(',') if ce.strip().isdigit()]

                    # Log expert selection
                    if SHOW_AGENT_INTERACTIONS:
                        agent_emoji_char = agent_emoji[agent_idx % len(agent_emoji)]
                        if chosen_expert_indices:
                            target_names = [f"Agent {idx} ({agent_emoji[idx-1 % len(agent_emoji)]} {medical_agents_list[idx-1].role})" for idx in chosen_expert_indices if 1 <= idx <= num_active_agents]
                            cprint(f"[EXPERT SELECTION] Agent {agent_idx+1} ({agent_emoji_char} {agent_instance_loop.role}) wants to talk to: {', '.join(target_names)}", 'magenta')
                            cprint(f"[SELECTION RESPONSE] {chosen_expert_response}", 'magenta')
                        else:
                            cprint(f"[EXPERT SELECTION] Agent {agent_idx+1} ({agent_emoji_char} {agent_instance_loop.role}) - No valid experts selected", 'yellow')

                    for target_expert_idx_chosen in chosen_expert_indices:
                        if 1 <= target_expert_idx_chosen <= num_active_agents:
                            target_agent_actual_idx = target_expert_idx_chosen - 1
                            specific_question_to_expert = agent_instance_loop.chat(f"Please remind your medical expertise and then leave your opinion/question for an expert you chose (Agent {target_expert_idx_chosen}. {medical_agents_list[target_agent_actual_idx].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert. Limit your response with no more than 200 words.") 
                            
                            source_emoji = agent_emoji[agent_idx % len(agent_emoji)]
                            target_emoji = agent_emoji[target_agent_actual_idx % len(agent_emoji)]
                            if SHOW_INTERACTION_TABLE:
                                print(f" Agent {agent_idx+1} ({source_emoji} {medical_agents_list[agent_idx].role}) -> Agent {target_expert_idx_chosen} ({target_emoji} {medical_agents_list[target_agent_actual_idx].role}) : {specific_question_to_expert}")
                            
                            # Log detailed interaction content
                            if SHOW_AGENT_INTERACTIONS:
                                cprint(f"[COMMUNICATION] Agent {agent_idx+1} ({source_emoji} {medical_agents_list[agent_idx].role}) -> Agent {target_expert_idx_chosen} ({target_emoji} {medical_agents_list[target_agent_actual_idx].role})", 'green')
                                cprint(f"[MESSAGE] {specific_question_to_expert}", 'green')
                            
                            interaction_log[round_name_str][turn_name_str][f'Agent {agent_idx+1}'][f'Agent {target_expert_idx_chosen}'] = specific_question_to_expert
                        else:
                            cprint(f"Agent {agent_idx+1} chose an invalid expert index: {target_expert_idx_chosen}", "yellow")
                
                    num_participated_this_turn += 1
                else:
                    if SHOW_INTERACTION_TABLE:
                        print(f" Agent {agent_idx+1} ({agent_emoji[agent_idx % len(agent_emoji)]} {agent_instance_loop.role}): \U0001f910")

            num_participated_this_round = num_participated_this_turn
            if num_participated_this_turn == 0:
                cprint(f"No agents chose to speak in {round_name_str}, {turn_name_str}. Moving to next round or finalizing.", "cyan")
                break
        
        if num_participated_this_round == 0 and r_idx > 1:
            cprint(f"No agents participated in {round_name_str} after initial opinions. Finalizing discussion.", "cyan")
            break

        if r_idx < num_rounds:
            next_round_opinions = {}
            for agent_idx_collect, agent_instance_collect in enumerate(medical_agents_list):
                opinion_prompt = f"Reflecting on the discussions in Round {r_idx}, what is your current answer/opinion on the question: {question}\n Limit your answer within 50 words" 
                response = agent_instance_collect.chat(opinion_prompt)
                next_round_opinions[agent_instance_collect.role.lower()] = response
            round_opinions_log[r_idx+1] = next_round_opinions
        
        current_round_final_answers = {}
        for agent_instance_final in medical_agents_list:
            response = agent_instance_final.chat(f"Now that you've interacted with other medical experts this round, remind your expertise and the comments from other experts and make your final answer to the given question for this round:\n{question}\n limit your answer within 50 words.")
            current_round_final_answers[agent_instance_final.role] = response
        final_answer_map = current_round_final_answers

    # Display interaction table only if enabled
    if SHOW_INTERACTION_TABLE:
        print('\nInteraction Log Summary Table')        
        myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i%len(agent_emoji)]})" for i in range(num_active_agents)])

        for i in range(1, num_active_agents + 1):
            row_data = [f"Agent {i} ({agent_emoji[(i-1)%len(agent_emoji)]})"]
            for j in range(1, num_active_agents + 1):
                if i == j:
                    row_data.append('')
                else:
                    i_to_j_spoke = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                                     for k in range(1, num_rounds + 1) if f'Round {k}' in interaction_log
                                     for l in range(1, num_turns + 1) if f'Turn {l}' in interaction_log[f'Round {k}'])
                    j_to_i_spoke = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                                     for k in range(1, num_rounds + 1) if f'Round {k}' in interaction_log
                                     for l in range(1, num_turns + 1) if f'Turn {l}' in interaction_log[f'Round {k}'])
                    
                    if not i_to_j_spoke and not j_to_i_spoke:
                        row_data.append(' ')
                    elif i_to_j_spoke and not j_to_i_spoke:
                        row_data.append(f'\u270B ({i}->{j})')
                    elif j_to_i_spoke and not i_to_j_spoke:
                        row_data.append(f'\u270B ({i}<-{j})')
                    elif i_to_j_spoke and j_to_i_spoke:
                        row_data.append(f'\u270B ({i}<->{j})')
            myTable.add_row(row_data)
            if i != num_active_agents:
                 myTable.add_row(['---' for _ in range(num_active_agents + 1)])
        print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model_to_use)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    if not final_answer_map:
        cprint("Warning: No final answers recorded from agents. Using initial opinions for moderation.", "yellow")
        final_answer_map = round_opinions_log[1]

    moderator_decision_dict = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote or synthesizing the best response. \nAgent Opinions:\n{final_answer_map}\n\nQuestion: {question} \nOnly respond with the correct option and answer, in this format: Answer: C) Example Answer. Do not include any explanations.", img_path=None)
    
    majority_vote_response = moderator_decision_dict.get(0.0, "Error: Moderator failed to provide a decision.")
    final_decision_output = {'majority_vote': majority_vote_response}

    print(f"{'\U0001F468\u200D\u2696\uFE0F'} moderator's final decision: {majority_vote_response}")

    # Calculate total token usage for this sample
    recruiter_usage = recruiter_agent.get_token_usage()
    sample_input_tokens += recruiter_usage['input_tokens']
    sample_output_tokens += recruiter_usage['output_tokens']
    
    for agent in medical_agents_list:
        agent_usage = agent.get_token_usage()
        sample_input_tokens += agent_usage['input_tokens']
        sample_output_tokens += agent_usage['output_tokens']
    
    moderator_usage = moderator.get_token_usage()
    sample_input_tokens += moderator_usage['input_tokens']
    sample_output_tokens += moderator_usage['output_tokens']
    
    # Include all summarizer agents created during rounds
    for summarizer_agent in summarizer_agents_list:
        summarizer_usage = summarizer_agent.get_token_usage()
        sample_input_tokens += summarizer_usage['input_tokens']
        sample_output_tokens += summarizer_usage['output_tokens']

    return final_decision_output, sample_input_tokens, sample_output_tokens

def process_advanced_query(question, model_to_use):
    # Reset token usage for this sample
    sample_input_tokens = 0
    sample_output_tokens = 0
    
    cprint("[STEP 1] Recruitment of Multidisciplinary Teams (MDTs)", 'yellow', attrs=['blink'])
    group_instances_list = []

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    recruiter_agent_mdt = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_to_use)
    recruiter_agent_mdt.chat(recruit_prompt)

    num_teams_to_form = 3
    num_agents_per_team = 3

    recruited_mdt_response = recruiter_agent_mdt.chat(f"Question: {question}\n\nYou should organize {num_teams_to_form} MDTs with different specialties or purposes and each MDT should have {num_agents_per_team} clinicians. Return your recruitment plan in JSON format with the following structure:\n\n{{\n  \"teams\": [\n    {{\n      \"team_id\": 1,\n      \"team_name\": \"Initial Assessment Team (IAT)\",\n      \"members\": [\n        {{\n          \"member_id\": 1,\n          \"role\": \"Otolaryngologist (ENT Surgeon) (Lead)\",\n          \"expertise_description\": \"Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\"\n        }},\n        {{\n          \"member_id\": 2,\n          \"role\": \"General Surgeon\",\n          \"expertise_description\": \"Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\"\n        }},\n        {{\n          \"member_id\": 3,\n          \"role\": \"Anesthesiologist\",\n          \"expertise_description\": \"Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\"\n        }}\n      ]\n    }}\n  ]\n}}\n\nYou must include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. Each team should have exactly {num_agents_per_team} members with one designated as Lead. Return only valid JSON without markdown code blocks or explanations.")

    # Clean and parse JSON response for MDT recruitment
    def clean_json_response(response_text):
        """Clean JSON response by removing markdown code blocks and extra formatting"""
        # Remove markdown code blocks
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith('```'):
            response_text = response_text[3:]   # Remove ```
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove trailing ```
        
        # Find JSON content between braces
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            response_text = response_text[start_idx:end_idx+1]
        
        return response_text.strip()

    try:
        cleaned_response = clean_json_response(recruited_mdt_response)
        mdt_data = json.loads(cleaned_response)
        teams = mdt_data.get('teams', [])
        
        for team_data in teams:
            team_name = team_data.get('team_name', f"Team {team_data.get('team_id', 'Unknown')}")
            members = team_data.get('members', [])
            
            print(f"Group {team_data.get('team_id', len(group_instances_list)+1)} - {team_name}")
            
            # Convert JSON member format to the expected format
            parsed_members = []
            for member in members:
                role = member.get('role', 'Unknown Role')
                expertise = member.get('expertise_description', 'No description available')
                parsed_members.append({
                    'role': role,
                    'expertise_description': expertise
                })
                # print(f" Member {member.get('member_id', len(parsed_members))} ({role}): {expertise}")
            # print()
            
            # Create Group instance
            group_instance_obj = Group(team_name, parsed_members, question, examplers=None, model_info=model_to_use)
            group_instances_list.append(group_instance_obj)
            
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        cprint(f"Warning: Failed to parse JSON MDT response: {e}. Falling back to text parsing.", "yellow")
        
        # Fallback to original text parsing method
        groups_text_list = [group_text.strip() for group_text in recruited_mdt_response.split("Group") if group_text.strip()]
        group_strings_list = ["Group " + group_text for group_text in groups_text_list]
        
        for i1, group_str_item in enumerate(group_strings_list):
            parsed_group_struct = parse_group_info(group_str_item)
            print(f"Group {i1+1} - {parsed_group_struct['group_goal']}")
            for i2, member_item in enumerate(parsed_group_struct['members']):
                print(f" Member {i2+1} ({member_item['role']}): {member_item['expertise_description']}")
            # print()

            # Create Group without examplers (no few-shot learning)
            group_instance_obj = Group(parsed_group_struct['group_goal'], parsed_group_struct['members'], question, examplers=None, model_info=model_to_use)
            group_instances_list.append(group_instance_obj)

    cprint("[STEP 2] MDT Internal Interactions and Assessments", 'yellow', attrs=['blink'])
    initial_assessments_list = []
    other_mdt_assessments_list = []
    final_review_decisions_list = []

    for group_obj in group_instances_list:
        group_goal_lower = group_obj.goal.lower()
        if 'initial' in group_goal_lower or 'iat' in group_goal_lower:
            cprint(f"Processing Initial Assessment Team: {group_obj.goal}", "cyan")
            init_assessment_text = group_obj.interact(comm_type='internal')
            initial_assessments_list.append([group_obj.goal, init_assessment_text])
        elif 'review' in group_goal_lower or 'decision' in group_goal_lower or 'frdt' in group_goal_lower:
            cprint(f"Processing Final Review/Decision Team: {group_obj.goal}", "cyan")
            decision_text = group_obj.interact(comm_type='internal')
            final_review_decisions_list.append([group_obj.goal, decision_text])
        else:
            cprint(f"Processing Specialist Team: {group_obj.goal}", "cyan")
            assessment_text = group_obj.interact(comm_type='internal')
            other_mdt_assessments_list.append([group_obj.goal, assessment_text])
    
    compiled_report_str = "[Initial Assessments]\n"
    for idx, init_assess_tuple in enumerate(initial_assessments_list):
        compiled_report_str += f"Team {idx+1} - {init_assess_tuple[0]}:\n{init_assess_tuple[1]}\n\n"
    
    compiled_report_str += "[Specialist Team Assessments]\n"
    for idx, assess_tuple in enumerate(other_mdt_assessments_list):
        compiled_report_str += f"Team {idx+1} - {assess_tuple[0]}:\n{assess_tuple[1]}\n\n"

    compiled_report_str += "[Final Review Team Decisions (if any before overall final decision)]\n"
    for idx, decision_tuple in enumerate(final_review_decisions_list):
        compiled_report_str += f"Team {idx+1} - {decision_tuple[0]}:\n{decision_tuple[1]}\n\n"

    cprint("[STEP 3] Final Decision from Overall Coordinator", 'yellow', attrs=['blink'])
    final_decision_prompt = f"""You are an experienced medical coordinator. Given the investigations and conclusions from multiple multidisciplinary teams (MDTs), please review them very carefully and return your final, consolidated answer to the medical query."""
    
    final_decision_agent = Agent(instruction=final_decision_prompt, role='Overall Coordinator', model_info=model_to_use)
    final_decision_agent.chat(final_decision_prompt)

    # Create JSON-formatted prompt similar to arbitrator pattern
    coordinator_prompt = f"""You are a medical coordinator. Review the following MDT investigations and conclusions and provide your final decision in JSON format:

Combined MDT Investigations and Conclusions:
{compiled_report_str}

Question: {question}

Analyze all MDT team assessments and provide your final decision in exactly this JSON format:

{{
  "analysis": "Your analysis of the MDT assessments and rationale for final decision in no more than 300 words",
  "final_answer": "X) Example Answer"
}}

**Requirements:**
- Consider all MDT team assessments in your analysis
- Final answer must correspond to one of the provided options
- Return ONLY the JSON, no other text
"""
    
    final_decision_dict_adv = final_decision_agent.temp_responses(coordinator_prompt, img_path=None)
    raw_final_response = final_decision_dict_adv.get(0.0, "Error: Final coordinator failed to provide a decision.")
    
    # Parse coordinator JSON response similar to arbitrator parsing
    try:
        json_match = re.search(r'\{[^{}]*"analysis"\s*:[^{}]*"final_answer"\s*:\s*"[^"]*"[^{}]*\}', raw_final_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            final_decision_dict = json.loads(json_str)
            final_response_str = final_decision_dict.get('final_answer', raw_final_response)
        else:
            # Fallback to raw response if JSON parsing fails
            final_response_str = raw_final_response
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Fallback to raw response if JSON parsing fails
        final_response_str = raw_final_response
    cprint(f"Overall Coordinated Final Decision: {final_response_str}", "green")
    
    # Calculate total token usage for this sample
    recruiter_usage = recruiter_agent_mdt.get_token_usage()
    sample_input_tokens += recruiter_usage['input_tokens']
    sample_output_tokens += recruiter_usage['output_tokens']
    
    # Track token usage from all group members (including all agents in each group)
    for group in group_instances_list:
        for member in group.members:
            member_usage = member.get_token_usage()
            sample_input_tokens += member_usage['input_tokens']
            sample_output_tokens += member_usage['output_tokens']
    
    final_agent_usage = final_decision_agent.get_token_usage()
    sample_input_tokens += final_agent_usage['input_tokens']
    sample_output_tokens += final_agent_usage['output_tokens']
    
    return {0.0: final_response_str}, sample_input_tokens, sample_output_tokens

