import os
import asyncio
import chainlit as cl
from dotenv import load_dotenv

from agents.document_agent import DocumentQAAgent
from prompts.prompt_template import (
    FILE_UPLOAD_MESSAGE,
    LITERATURE_AGENT_DESCRIPTION,
    DOCUMENT_AGENT_DESCRIPTION
)
from orchestrator.sk_router_planner import multi_agent_dispatch_stream

# Load env
load_dotenv(override=True)

SEARCH_AGENT = "search"
DOCUMENT_AGENT = "document"


@cl.set_chat_profiles
async def chat_profiles(current_user: cl.User):
    return [
        cl.ChatProfile(
            name="Search Agent",
            markdown_description=LITERATURE_AGENT_DESCRIPTION,
            icon="https://cdn-icons-png.flaticon.com/512/7641/7641727.png",
        ),
        cl.ChatProfile(
            name="Document Agent",
            markdown_description=DOCUMENT_AGENT_DESCRIPTION,
            icon="https://cdn-icons-png.flaticon.com/512/4725/4725970.png",
        ),
    ]


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    cl.user_session.set("active_documents", [])

    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Search Agent":
        cl.user_session.set("current_agent", SEARCH_AGENT)

    elif chat_profile == "Document Agent":
        cl.user_session.set("current_agent", DOCUMENT_AGENT)

        try:
            document_qa_agent = DocumentQAAgent()
            cl.user_session.set("document_qa_agent", document_qa_agent)

            files = await cl.AskFileMessage(
                content=FILE_UPLOAD_MESSAGE,
                accept=[
                    "application/pdf",
                    "text/plain",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ],
                max_size_mb=50,
                max_files=10,
                timeout=180
            ).send()

            if files:
                await process_files(files, document_qa_agent)

        except Exception as e:
            await cl.Message(content=f"❌ Init Error: {str(e)}").send()


async def process_files(files, agent):
    msg = cl.Message(content="Processing files...\n")
    await msg.send()

    for i, file in enumerate(files):
        try:
            file_extension = file.name.split('.')[-1].lower()
            
            chunks = agent.process_document(
                file.path,
                file_extension,
                file.name
            )

            msg.content += f"\n✅ {file.name} ({chunks} chunks)"
            await msg.update()

        except Exception as e:
            msg.content += f"\n❌ {file.name}: {str(e)}"
            await msg.update()


@cl.on_message
async def main(message: cl.Message):
    current_agent = cl.user_session.get("current_agent")

    if current_agent == SEARCH_AGENT:
        await handle_search(message)

    elif current_agent == DOCUMENT_AGENT:
        await handle_document(message)


async def handle_search(message):
    msg = cl.Message(content="Thinking...")
    await msg.send()

    full = ""

    try:
        async for token in multi_agent_dispatch_stream(message.content):
            if token and token != "⏳ Thinking...":
                if not full:
                    msg.content = ""
                    await msg.update()

                full += token
                await msg.stream_token(token)
                
        if full:
            await msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


async def handle_document(message):
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent = cl.user_session.get("document_qa_agent")

    if not agent:
        msg.content = "❌ Reload app"
        await msg.update()
        return

    full = ""

    try:
        async for token in agent.run_document_agent_stream(message.content):
            if token:
                if not full:
                    msg.content = ""
                    await msg.update()

                full += token
                await msg.stream_token(token)
                
        if full:
            await msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()


@cl.on_chat_end
async def end():
    agent = cl.user_session.get("document_qa_agent")

    if agent:
        agent.cleanup()


if __name__ == "__main__":
    import chainlit.cli as cli
    cli.run_chainlit(__file__)