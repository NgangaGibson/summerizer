



import streamlit as st
import whisper
import torch
from transformers import pipeline
from gtts import gTTS
import tempfile
import os

st.set_page_config(
    page_title="TL;DR Audio Summarizer",
    page_icon="🎧"
)

st.title("🎧 TL;DR Audio Summarizer")
st.write("Upload an audio file to get a transcript, short summary, and summary audio.")

uploaded_file = st.file_uploader(
    "Choose an audio file from File Explorer",
    type=["mp3", "wav", "m4a", "mpeg", "mpga", "webm", "ogg"]
)

summary_length = st.selectbox(
    "Choose summary length",
    ["Short", "Medium", "Detailed"]
)

if uploaded_file is not None:
    st.success(f"Selected file: {uploaded_file.name}")

    if st.button("Transcribe and Summarize"):
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as temp_audio:
                temp_audio.write(uploaded_file.getvalue())
                audio_path = temp_audio.name

            with st.spinner("Loading speech recognition model..."):
                speech_model = whisper.load_model("base")

            with st.spinner("Transcribing audio..."):
                result = speech_model.transcribe(audio_path)
                transcript = result["text"].strip()

            st.subheader("📝 Audio Transcript")
            st.write(transcript)

            if not transcript:
                st.error("No speech was detected in this audio.")
                st.stop()

            max_lengths = {
                "Short": 60,
                "Medium": 120,
                "Detailed": 220
            }

            with st.spinner("Summarizing the transcript..."):
                summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=0 if torch.cuda.is_available() else -1
                )

                chunks = [
                    transcript[i:i + 2500]
                    for i in range(0, len(transcript), 2500)
                ]

                partial_summaries = []

                for chunk in chunks:
                    if len(chunk.split()) < 35:
                        partial_summaries.append(chunk)
                    else:
                        output = summarizer(
                            chunk,
                            max_length=max_lengths[summary_length],
                            min_length=20 if summary_length == "Short" else 30,
                            do_sample=False,
                            truncation=True
                        )
                        partial_summaries.append(output[0]["summary_text"])

                summary = " ".join(partial_summaries)

            st.subheader("✨ TL;DR Summary")
            st.write(summary)

            with st.spinner("Generating summary audio..."):
                tts = gTTS(text=summary, lang="en")
                audio_output = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                )
                audio_output.close()
                tts.save(audio_output.name)

            st.subheader("🔊 Listen to the Summary")
            with open(audio_output.name, "rb") as audio_file:
                audio_bytes = audio_file.read()

            st.audio(audio_bytes, format="audio/mp3")

            st.download_button(
                "Download Summary Audio",
                data=audio_bytes,
                file_name="summary.mp3",
                mime="audio/mpeg"
            )

            st.download_button(
                "Download Text Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

            st.download_button(
                "Download Full Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")

        finally:
            if "audio_path" in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
