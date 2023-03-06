from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import (
    ConversationalBufferWindowMemory,
)

from chatllama.langchain_modules.prompt_templates import (
    PERSON_CHATBOT_TEMPLATE,
    AI_CHATBOT_TEMPLATE,
)
import logging

CONVERSATION_LENGTH = 20


def create_conversation(human_agent: LLMChain, bot_agent: LLMChain):
    conversation = []
    chatbot_output = ""
    for i in range(CONVERSATION_LENGTH):
        # Human agent goes first
        human_output = human_agent.run(chatbot_input=chatbot_output)
        conversation.append(f"Human: {human_output}")
        chatbot_output = bot_agent.run(human_input=human_output)
        conversation.append(f"AI: {chatbot_output}")
    return "\n".join(conversation)


def build_agents():
    # be aware that too long completions will not fit the sequence length
    # of possible critic or reward models ...
    llm = OpenAI(max_tokens=2048, temperature=0.7)
    human_template = PromptTemplate(**PERSON_CHATBOT_TEMPLATE)
    human_agent = LLMChain(
        llm=llm,
        prompt=human_template,
        memory=ConversationalBufferWindowMemory(k=4),
    )
    bot_template = PromptTemplate(**AI_CHATBOT_TEMPLATE)
    bot_agent = LLMChain(
        llm=llm,
        prompt=bot_template,
        memory=ConversationalBufferWindowMemory(k=4),
    )
    return human_agent, bot_agent

def filters(line):
    """
        This function filters out lines that are not part of a conversation, like missed calls
        or images imitted
    """
    if ("Missed voice call" in line) or  ("Missed video call" in line):
        return False
    elif ("image omitted" in line) or ("video omitted" in line):
        return False
    elif ("https://" in line):
        return False
    elif ("[" not in line):
        return False
    else:
        return True

def consolidate_lines(lines):
    """
        This function takes in a chat thread and consolidates multiple lines from one person into one entry
    """
    counter, lines_len= 0, len(lines)
    new_lines = []
    previous_person = ""
    consolidated_line = ""
    while counter < lines_len:
        this_person = lines[counter].split("] ")[1].split(": ")[0]
        content = lines[counter].split(": ")[1][:-1]
        if this_person==previous_person:
            consolidated_line += " " + content
        if this_person!=previous_person:
            if previous_person == "ToFu":
               new_lines.append("AI: " + consolidated_line)
            else:
               new_lines.append("Human: " + consolidated_line)
            previous_person = this_person
            consolidated_line = content
        counter += 1
    return new_lines

def get_all_conversations(lines, num_conversations):
    """
        I am just doing a brute force with a sliding window of 20
    """
    all_conversations = []
    i,j=0,20
    while j<len(lines) and len(all_conversations)<num_conversations:
        if lines[i].startswith("Human"):
           this_conversation = lines[i:i+20]
           all_conversations.append("\n".join(this_conversation))
        i+=1
        j+=1
    return all_conversations
    

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_conversations", type=int, default=1000)
    parser.add_argument("--output_file", type=str, default="conversations.txt")
    parser.add_argument("--from_local_chats", type=bool, default=True)
    parser.add_argument("--chats_file", type=str, default="/root/chats/_chat_1.txt"  )
    args = parser.parse_args()
    conversations = []
    # it looks like we want a conversation length of 20 from human and AI, we
    # need to remove some bad lines like missed your call, re
    if args.from_local_chats:
        with open(args.chats_file, "r") as f:
            lines = f.readlines()
        lines = list(filter(filters, lines))
        lines = consolidate_lines(lines)
        print(lines)
        conversations = get_all_conversations(lines, args.num_conversations)
    else:
        for conv in range(args.num_conversations):
            human_agent, bot_agent = build_agents()
            conversation = create_conversation(human_agent, bot_agent)
            conversations.append(conversation)
    with open(args.output_file, "w") as f:
        f.write("\n\nNEW CONVERSATION\n\n".join(conversations))

if __name__ == "__main__":
    main()