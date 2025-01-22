import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import time 
from pathlib import Path
import tempfile
import os
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title = "Video Summarizer",
    page_icon = "📹",
    layout = "centered",
)

st.title("Your AI Video Summarizer")
st.header("Summarize your video in a few seconds!")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="video_summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

## Initialize the agent
agent = initialize_agent()

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv", "webm"],
                              help="Upload a video file to summarize")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What would you like to know about this video?",
        placeholder="Ask a question or type a keyword",
        help="Ask a question or type a keyword to get a summary of the video",
    )

    if st.button("Summarize", key="Summarize"):
        if not user_query:
            st.error("Please provide a query to summarize the video")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)

                    while processed_video.state.name == "PROCESSING":
                        time.sleep(5)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following qeury using video insghts and supplementary web search
                        {user_query}

                        Provide a detailed, user-friendly and actionable response.

                        """
                    )

                    response = agent.run(analysis_prompt, videos = [processed_video])

                st.subheader("Analysis Result")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"An error occured during analysis: {e}")

            finally:
                Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Uplaod a video file to begin analysis")

    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
                

        

