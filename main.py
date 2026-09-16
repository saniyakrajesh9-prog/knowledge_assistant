import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain.tools import tool

from rag import get_retriever


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CHECK API KEY
# ============================================================

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please add it to your .env file."
    )


# ============================================================
# SETUP RETRIEVER
# ============================================================

retriever, num_docs = get_retriever()


# ============================================================
# LLM - OPENAI
# ============================================================

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================
# TOOLS
# ============================================================

@tool
def document_search(query: str) -> str:
    """
    Search information from the uploaded documents.
    Use this tool when the user asks questions about the documents.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No information found in the document."

    results = []

    for doc in docs:
        results.append(doc.page_content)

    return "\n\n".join(results)


@tool
def system_datetime() -> str:
    """
    Get the current date and time.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# BIND TOOLS
# ============================================================

tools = [
    document_search,
    system_datetime,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CONVERSATION HISTORY
# ============================================================

chat_history = []


# ============================================================
# GET TEXT FROM RESPONSE
# ============================================================

def get_text(response):

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Content blocks
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

        return "".join(text_parts)

    return str(content)


# ============================================================
# ASK AI
# ============================================================

def ask(query):

    # Keep last 10 messages
    messages = chat_history[-10:] + [
        HumanMessage(content=query)
    ]

    # ========================================================
    # FIRST LLM CALL
    # ========================================================

    response = llm_with_tools.invoke(messages)

    # ========================================================
    # TOOL CALL
    # ========================================================

    if response.tool_calls:

        # Add AI response containing tool calls
        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            # ------------------------------------------------
            # DOCUMENT SEARCH
            # ------------------------------------------------

            if tool_name == "document_search":

                result = document_search.invoke(
                    tool_call["args"]
                )

            # ------------------------------------------------
            # DATE / TIME
            # ------------------------------------------------

            elif tool_name == "system_datetime":

                result = system_datetime.invoke(
                    tool_call["args"]
                )

            # ------------------------------------------------
            # UNKNOWN TOOL
            # ------------------------------------------------

            else:

                result = "Unknown tool."

            # ------------------------------------------------
            # ADD TOOL RESULT
            # ------------------------------------------------

            from langchain.messages import ToolMessage

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

        # ====================================================
        # SECOND LLM CALL
        # ====================================================

        response = llm_with_tools.invoke(messages)

    # ========================================================
    # GET ANSWER
    # ========================================================

    answer = get_text(response)

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    chat_history.append(
        HumanMessage(content=query)
    )

    chat_history.append(response)

    return answer


# ============================================================
# START PROGRAM
# ============================================================

print("==================================")
print("SMART AI KNOWLEDGE ASSISTANT")
print("==================================")

print(f"Documents loaded: {num_docs}")

print("Type 'exit' to quit.")


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    user = input("\nYou: ")

    if user.lower().strip() in ["exit", "quit"]:

        print("Goodbye!")

        break

    if not user.strip():
        continue

    try:

        answer = ask(user)

        print("\nAI:", answer)

    except Exception as e:

        print("\nError:", str(e))
