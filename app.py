import tempfile
import os

from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("음성 번역기 (한국어 → 일본어 음성)")

uploaded_file = st.file_uploader("음성을 업로드하세요 (예: mp3, wav)", type=["mp3", "wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.audio(temp_path, format="audio/mp3")
#sound to txt
    with open(temp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    original_text = transcript.text
    st.subheader("인식된 텍스트")
    st.write(original_text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Translate the following text into Japanese."},
            {"role": "user", "content": original_text}
        ]
    )
#txt to foreign txt
    translated_text = response.choices[0].message.content
    st.subheader("일본어 번역")
    st.write(translated_text)

    speech_file_path = os.path.join(tempfile.gettempdir(), "japanese_output.mp3")
#txt to audio
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=translated_text
    ) as response:
        response.stream_to_file(speech_file_path)

    st.subheader("일본어 음성 출력")
    st.audio(speech_file_path, format="audio/mp3")


